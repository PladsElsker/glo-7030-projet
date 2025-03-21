from typing import Final

from preprocessing.bg_changer.bg_changer_processor import OpenCVBackgroundChanger
from preprocessing.resize_videos.resizer_processor import FFmpegProcessor
from preprocessing.video_processor import VideoProcessor

PATHS: Final[dict[str, str]] = {
    "data_root": "data",
    "raw_videos_dir": "raw_videos",
    "FFmpegProcessor_output_dir_template": "{size}x{size}",
    "OpenCVBackgroundChanger_output_dir_template": "{input_dir_name}_no_green_bg",
}

PROCESSING_CONFIG: Final[dict[str, bool | int]] = {
    "batch_size": 1,
    "verbose": True,
}

SUPPORTED_EXTENSIONS: Final[list[str]] = [".mp4", ".avi", ".mov", ".mkv"]

DATASET_SPLITS: Final[list[str]] = ["train", "val", "test"]
DATASET_VIEWS: Final[list[str]] = ["rgb_front", "rgb_side"]

PREPROCESSING_TYPES: list[str] = ["resize_videos", "change_bg"]
PREPROCESSOR_TYPES: dict[str, VideoProcessor] = {
    "resize_videos": FFmpegProcessor(),
    "change_bg": OpenCVBackgroundChanger(),
}

PREPROCESSING_STARTING_MSG: dict[str, str] = {
    "resize_videos": "Starting video(s) resizing process with size {size}x{size}",
    "change_bg": "Starting video(s) changing background process with size {size}",
}
