import random
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from loguru import logger

from preprocessing.video_processor import VideoProcessor

from .config import DEFAULT_CONFIG, LOWER_GREEN, UPPER_GREEN


class AutoSaveMode(Enum):
    YES = True
    NO = False


class BackgroundChangerProcessor(VideoProcessor):
    @abstractmethod
    def change_background(self, input_path: Path, output_path: Path, config: dict[str, AutoSaveMode | Path]) -> bool | np.ndarray:
        """
        Replace green background by a random video or picture.

        :param input_path: Path to the input video file.
        :param output_path: Path to save the processed video.
        :param config: Dictionary with keys {"background_dir_path", "is_autosave_activated"}.
        :return: True if successful, False otherwise.
        """

    def process(self, input_path: Path, output_path: Path, *_: tuple[Any, ...]) -> bool | np.ndarray:
        return self.change_background(input_path, output_path, DEFAULT_CONFIG)


class OpenCVBackgroundChanger(BackgroundChangerProcessor):
    def __init__(self, size: Optional[int] = None, autosave_mode: AutoSaveMode = AutoSaveMode.YES) -> None:
        self.bg_color_range = [LOWER_GREEN, UPPER_GREEN]
        self.size = size
        self.fps = 30
        self.autosave_mode = autosave_mode

    @staticmethod
    def __select_background__(backgrounds_dir_path: Path) -> Path:
        all_backgrounds = list(backgrounds_dir_path.glob("*.jpg"))
        return random.choice(all_backgrounds)

    def change_background(self, input_path: Path, output_path: Path, config: dict[str, AutoSaveMode | Path]) -> bool | np.ndarray:

        if "background_dir_path" not in config:
            logger.error("Background directory must be specified.")

        background_dir_path = Path(config["background_dir_path"])

        try:
            if not input_path.exists():
                logger.error(f"Video not found: {input_path}")
                return False

            if not background_dir_path.exists():
                logger.error(f"Background image not found: {background_dir_path}")
                return False

            video = cv2.VideoCapture(str(input_path))
            background = cv2.imread(str(self.__select_background__(background_dir_path)))

            if not video.isOpened():
                logger.error(f"Cannot open video: {input_path}")
                return False

            ret, frame = video.read()
            if not ret:
                logger.error(f"Cannot read video: {input_path}")
                return False

            frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            kernel = np.ones((5, 5), np.uint8)
            frames_list = []
            background = cv2.resize(background, (frame_width, frame_height))

            while ret:
                frame = cv2.resize(frame, (frame_width, frame_height))
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                mask = cv2.inRange(hsv, self.bg_color_range[0], self.bg_color_range[1])
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.dilate(mask, kernel, iterations=1)
                mask = cv2.GaussianBlur(mask, (1, 1), 0)

                mask_inv = cv2.bitwise_not(mask)
                human = cv2.bitwise_and(frame, frame, mask=mask_inv)
                background_part = cv2.bitwise_and(background, background, mask=mask)

                final_result = cv2.add(human, background_part)
                frames_list.append(final_result)

                ret, frame = video.read()

            video.release()
            cv2.destroyAllWindows()

            return self._save(np.array(frames_list), output_path) if self.autosave_mode.value else np.array(frames_list)

        except ValueError as e:
            logger.error(f"Error processing {input_path}: {e}")
            return False

    def _save(self, video_frames: np.ndarray, output_path: Path) -> bool:
        if video_frames is None or video_frames.size == 0:
            logger.error("No frames to save.")
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        num_frames, height, width, _ = video_frames.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))

        try:
            canal_size = 4
            for frame in video_frames:
                frame_to_write = frame
                if frame.shape[2] == canal_size:
                    frame_to_write = frame[:, :, :3]
                out.write(frame_to_write)

            out.release()
            logger.info(f"Video saved: {output_path}")

        except ValueError as e:
            logger.error(f"Error saving {output_path}: {e}")
            return False
        return True

    def deactivate_autosave(self) -> None:
        self.autosave_mode = AutoSaveMode.NO


if __name__ == "__main__":
    processor = OpenCVBackgroundChanger()
