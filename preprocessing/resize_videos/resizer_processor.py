import subprocess
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from preprocessing.video_processor import VideoProcessor


class ResizerProcessor(VideoProcessor):
    @abstractmethod
    def resize_video(self, input_path: Path, output_path: Path, config: dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def resize_video_from_nparray(self, input_array: np.ndarray, output_path: Path, config: dict[str, Any]) -> bool:
        pass

    def process(self, input_path: Path, output_path: Path, config: dict[str, Any]) -> bool:
        return self.resize_video(input_path, output_path, config)


class FFmpegProcessor(ResizerProcessor):
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

    def resize_video_from_nparray(self, input_array: np.ndarray, output_path: Path, config: dict[str, Any]) -> bool:
        array_dim, nb_canal = 4, 3

        if input_array.ndim != array_dim or input_array.shape[3] != nb_canal:
            err_msg = "input_array must have shape: frames, height, width, 3"
            raise ValueError(err_msg)

        num_frames, height, width, _ = input_array.shape

        scale_crop_filter = f"scale='if(gt(a,1),-1,{config['width']})':'if(gt(a,1),{config['height']},-1)',crop={config['width']}:ih:(iw-ow)/2:0"

        cmd = [
            "ffmpeg",
            "-y" if config.get("overwrite", False) else "-n",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(config.get("fps", 30)),
            "-i",
            "-",  # Input from stdin
            "-vf",
            scale_crop_filter,
            "-an",  # No audio
            "-vcodec",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            str(output_path),
        ]

        try:
            input_array = input_array[..., ::-1]
            result = subprocess.run(cmd, input=input_array.tobytes(), capture_output=True, check=False)  # noqa: S603

        except (subprocess.SubprocessError, OSError) as e:
            logger.error(f"Exception during ffmpeg execution from array: {e}")
            return False

        else:
            if result.returncode != 0:
                logger.error(f"Error resizing video from array: {result.stderr}")
            return result.returncode == 0
