import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from preprocessing.config.video_resize_config import (
    PATHS, get_resize_config, PROCESSING_CONFIG, 
    SUPPORTED_EXTENSIONS, get_clip_directories, get_output_dir
)

def setup_logger():
    """Configure the logger"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

def create_output_dirs(output_path):
    """Create necessary output directories"""
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created/verified: {output_path}")

def resize_video(input_path, output_path, resize_config):
    """Resize a video using ffmpeg"""
    try:
        # Build ffmpeg command
        if resize_config['maintain_aspect']:
            # Use scale with force_original_aspect_ratio=decrease to maintain aspect ratio
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-vf', f"scale={resize_config['width']}:{resize_config['height']}:force_original_aspect_ratio=decrease",
                '-c:a', 'copy',  # Copy audio without modification
                '-y' if PROCESSING_CONFIG['overwrite'] else '-n',  # Overwrite or not existing files
                str(output_path)
            ]
        else:
            # Force exact dimensions
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-vf', f"scale={resize_config['width']}:{resize_config['height']}",
                '-c:a', 'copy',  # Copy audio without modification
                '-y' if PROCESSING_CONFIG['overwrite'] else '-n',  # Overwrite or not existing files
                str(output_path)
            ]
        
        # Execute command
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Error during resizing of {input_path}: {result.stderr}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Exception during resizing of {input_path}: {str(e)}")
        return False

def process_video_file(input_path, output_path, resize_config):
    """Process a single video file"""
    input_path = Path(input_path)
    if not input_path.exists():
        logger.warning(f"File not found: {input_path}")
        return False

    output_path = Path(output_path)
    
    # Create parent directory if necessary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not PROCESSING_CONFIG['overwrite']:
        logger.info(f"File already exists, skipped: {output_path}")
        return True

    logger.info(f"Processing: {input_path}")
    success = resize_video(input_path, output_path, resize_config)
    
    if success:
        logger.success(f"Video successfully resized: {output_path}")
    else:
        logger.error(f"Resizing failed: {input_path}")
    
    return success

def process_directory(clip_dir, size, square=True):
    """Process all video files in a directory"""
    # Get resize configuration
    resize_config = get_resize_config(size, square)
    
    # Construct paths
    raw_videos_path = Path(clip_dir) / PATHS['raw_videos_dir']
    output_path = Path(clip_dir) / get_output_dir(size)
    
    if not raw_videos_path.exists():
        logger.error(f"Raw videos directory not found: {raw_videos_path}")
        return
    
    # Create output directory
    create_output_dirs(output_path)
    
    # Find all video files
    video_files = []
    for ext in SUPPORTED_EXTENSIONS:
        video_files.extend(raw_videos_path.glob(f"*{ext}"))
    
    if not video_files:
        logger.warning(f"No video files found in {raw_videos_path}")
        return
    
    logger.info(f"Found {len(video_files)} video files to process in {raw_videos_path}")
    
    # Process videos in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=PROCESSING_CONFIG['batch_size']) as executor:
        # Submit all tasks
        future_to_video = {
            executor.submit(process_video_file, video_file, output_path / video_file.name, resize_config): video_file 
            for video_file in video_files
        }
        
        # Process results as they complete
        for future in as_completed(future_to_video):
            video_file = future_to_video[future]
            try:
                success = future.result()
                if not success:
                    logger.error(f"Failed to process {video_file}")
            except Exception as e:
                logger.error(f"Error processing {video_file}: {str(e)}")
    
    logger.info(f"Processing completed for {clip_dir}")

def process_all_datasets(size, square=True):
    """Process all video datasets"""
    data_root = Path(PATHS['data_root'])
    clip_dirs = get_clip_directories()
    
    for dataset in clip_dirs:
        dataset_path = data_root / dataset
        if not dataset_path.exists():
            logger.warning(f"Dataset directory not found: {dataset_path}")
            continue
            
        logger.info(f"Processing dataset: {dataset}")
        process_directory(dataset_path, size, square)
        logger.info(f"Completed processing dataset: {dataset}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Resize videos to specified dimensions')
    parser.add_argument('--size', type=int, default=224,
                      help='Target size for width and height (default: 224)')
    parser.add_argument('--data', type=str, help='Specific directory to process')
    parser.add_argument('--all', action='store_true',
                      help='Process all clip directories')
    parser.add_argument('--overwrite', action='store_true',
                      help='Overwrite existing files')
    parser.add_argument('--not-square', action='store_true',
                      help='Maintain original aspect ratio (default: force square)')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Update configuration based on arguments
    PROCESSING_CONFIG['overwrite'] = args.overwrite
    
    setup_logger()
    logger.info(f"Starting video resizing process with size {args.size}x{args.size}")
    if args.not_square:
        logger.info("Maintaining original aspect ratio")
    else:
        logger.info("Forcing square output")
    
    if args.data:
        # Process specific directory
        data_path = Path(args.data)
        if not data_path.exists():
            logger.error(f"Directory not found: {data_path}")
            return
        logger.info(f"Processing specific directory: {data_path}")
        process_directory(data_path, args.size, not args.not_square)
    elif args.all:
        # Process all clip directories
        logger.info("Processing all clip directories")
        process_all_datasets(args.size, not args.not_square)
    else:
        logger.error("Please specify either --data <directory> or --all")
        return
    
    logger.info("Process completed")

if __name__ == "__main__":
    main() 