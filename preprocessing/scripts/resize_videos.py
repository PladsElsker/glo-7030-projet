import os
import subprocess
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import sys
import click

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
        if resize_config['maintain_aspect']:
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-vf', f"scale={resize_config['width']}:{resize_config['height']}:force_original_aspect_ratio=decrease",
                '-c:a', 'copy', 
                '-y' if PROCESSING_CONFIG['overwrite'] else '-n', 
                str(output_path)
            ]
        else:
            # Force exact dimensions
            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-vf', f"scale={resize_config['width']}:{resize_config['height']}",
                '-c:a', 'copy', 
                '-y' if PROCESSING_CONFIG['overwrite'] else '-n',
                str(output_path)
            ]
        
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
    resize_config = get_resize_config(size, square)
    
    raw_videos_path = Path(clip_dir) / PATHS['raw_videos_dir']
    output_path = Path(clip_dir) / get_output_dir(size)
    
    if not raw_videos_path.exists():
        logger.error(f"Raw videos directory not found: {raw_videos_path}")
        return
    
    create_output_dirs(output_path)
    
    # Find all video files
    video_files = []
    for ext in SUPPORTED_EXTENSIONS:
        video_files.extend(raw_videos_path.glob(f"*{ext}"))
    
    if not video_files:
        logger.warning(f"No video files found in {raw_videos_path}")
        return
    
    logger.info(f"Found {len(video_files)} video files to process in {raw_videos_path}")
    
    # Process each video file
    for video_file in video_files:
        output_file = output_path / video_file.name
        process_video_file(video_file, output_file, resize_config)
    
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

@click.command()
@click.option('--size', default=224, help='Target size for width and height (default: 224)')
@click.option('--data', help='Specific directory to process')
@click.option('--all', is_flag=True, help='Process all clip directories')
@click.option('--overwrite', is_flag=True, help='Overwrite existing files')
@click.option('--not-square', is_flag=True, help='Maintain original aspect ratio (default: force square)')
def main(size, data, all, overwrite, not_square):
    """Resize videos to specified dimensions."""
    PROCESSING_CONFIG['overwrite'] = overwrite
    
    setup_logger()
    logger.info(f"Starting video resizing process with size {size}x{size}")
    if not_square:
        logger.info("Maintaining original aspect ratio")
    else:
        logger.info("Forcing square output")
    
    if data:
        data_path = Path(data)
        if not data_path.exists():
            logger.error(f"Directory not found: {data_path}")
            return
        logger.info(f"Processing specific directory: {data_path}")
        process_directory(data_path, size, not not_square)
    elif all:
        logger.info("Processing all clip directories")
        process_all_datasets(size, not not_square)
    else:
        logger.error("Please specify either --data <directory> or --all")
        return
    
    logger.info("Process completed")

if __name__ == "__main__":
    main() 