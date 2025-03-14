"""
Configuration for video resizing
"""

PATHS = {
    'data_root': 'data',
    'raw_videos_dir': 'raw_videos',
    'output_dir_template': '{size}x{size}',  
}

def get_resize_config(size=224):
    return {
        'width': size,    
        'height': size,     
    }

PROCESSING_CONFIG = {
    'batch_size': 1,  # Number of videos to process in parallel
    'overwrite': False,
    'verbose': True,
}

SUPPORTED_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']

DATASET_SPLITS = ['train', 'val', 'test']
DATASET_VIEWS = ['rgb_front', 'rgb_side']

def get_clip_directories():
    clip_dirs = []
    for split in DATASET_SPLITS:
        for view in DATASET_VIEWS:
            clip_dirs.append(f"{split}_{view}_clips")
    return clip_dirs

def get_output_dir(size):
    return PATHS['output_dir_template'].format(size=size)