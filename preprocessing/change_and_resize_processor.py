from pathlib import Path
from typing import TYPE_CHECKING, Any

from .bg_changer.bg_changer_processor import OpenCVBackgroundChanger
from .resize_videos.resizer_processor import FFmpegProcessor
from .video_processor import VideoProcessor

if TYPE_CHECKING:
    import numpy as np


class ChangeAndResizeProcessor(VideoProcessor):
    def __init__(self) -> None:
        self.ffmpeg_processor = FFmpegProcessor()
        self.background_processor = OpenCVBackgroundChanger()

    def process(self, input_path: Path, output_path: Path, config: dict[str, Any]) -> bool:
        video_array: np.ndarray = self.background_processor.process(input_path, output_path)
        return self.ffmpeg_processor.resize_video_from_nparray(video_array, output_path, config)
