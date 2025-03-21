from pathlib import Path
from typing import Any

from bg_changer.bg_changer_processor import OpenCVBackgroundChanger
from resize_videos.resizer_processor import FFmpegProcessor
from video_processor import VideoProcessor


class ChangeAndResizeProcessor(VideoProcessor):
    def __init__(self) -> None:
        self.ffmpeg_processor = FFmpegProcessor()
        self.background_processor = OpenCVBackgroundChanger()

    def process(self, input_path: Path, output_path: Path, config: dict[str, Any]) -> bool:
        self.background_processor.process(input_path, output_path)
        return self.ffmpeg_processor.process(input_path, output_path, config)
