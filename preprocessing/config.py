from preprocessing.bg_changer.bg_changer_processor import OpenCVBackgroundChanger
from preprocessing.resize_videos.resizer_processor import FFmpegProcessor
from preprocessing.video_processor import VideoProcessor

PREPROCESSING_TYPES: list[str] = ["resize_videos", "change_bg"]
PREPROCESSOR_TYPES: dict[str, VideoProcessor] = {
    "resize_videos": FFmpegProcessor(),
    "change_bg": OpenCVBackgroundChanger(),
}

PREPROCESSING_STARTING_MSG: dict[str, str] = {
    "resize_videos": "Starting video(s) resizing process with size {size}x{size}",
    "change_bg": "Starting video(s) changing background process with size {size}",
}
