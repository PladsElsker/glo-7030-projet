"""
Configuration for video resizing
"""

# Directory paths
PATHS = {
    'data_root': 'data',           # Root directory containing all video datasets
    'raw_videos_dir': 'raw_videos', # Subdirectory containing raw videos
    'output_dir_template': '{size}x{size}',  # Template for output directory name
}

def get_resize_config(size=224, square=True):
    """Get resize configuration with specified size"""
    return {
        'width': size,      # Target width
        'height': size,     # Target height
        'maintain_aspect': not square,  # If True, maintains aspect ratio (default: False)
    }

# Processing configuration
PROCESSING_CONFIG = {
    'batch_size': 10,  # Number of videos to process in parallel
    'overwrite': False,  # If True, overwrites existing files
    'verbose': True,    # Displays progress information
}

# Supported video file extensions
SUPPORTED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']

# Dataset splits and views
DATASET_SPLITS = ['train', 'val', 'test']
DATASET_VIEWS = ['rgb_front', 'rgb_side']

def get_clip_directories():
    """Generate list of all clip directories"""
    clip_dirs = []
    for split in DATASET_SPLITS:
        for view in DATASET_VIEWS:
            clip_dirs.append(f"{split}_{view}_clips")
    return clip_dirs

def get_output_dir(size):
    """Get output directory name based on size"""
    return PATHS['output_dir_template'].format(size=size) 