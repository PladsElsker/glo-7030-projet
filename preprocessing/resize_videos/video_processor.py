import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger


class VideoProcessor(ABC):
    @abstractmethod
    def resize_video(self, input_path: Path, output_path: Path, config: dict[str, Any]) -> bool:
        pass


class FFmpegProcessor(VideoProcessor):
    def resize_video(self, input_path: Path, output_path: Path, config: dict[str, Any]) -> bool:
        try:
            cmd = [
                "ffmpeg",
                "-i",
                str(input_path),
                "-vf",
                f"scale='if(gt(a,1),-1,{config['width']})':'if(gt(a,1),{config['height']},-1)',crop={config['width']}:ih:(iw-ow)/2:0",
                "-c:a",
                "copy",
                "-y" if config.get("overwrite", False) else "-n",
                str(output_path),
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
            except (subprocess.SubprocessError, OSError) as e:
                logger.error(f"Exception during ffmpeg execution: {e}")
                return False
            else:
                if result.returncode != 0:
                    logger.error(f"Error during resizing of {input_path}: {result.stderr}")
                return result.returncode == 0
        except (ValueError, KeyError) as e:
            logger.error(f"Configuration error during resizing of {input_path}: {e}")
            return False
