import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class BackgroundChanger:
    def __init__(
        self,
        input_directory: Path,
        backgrounds_directory: Path,
        output_directory: Path,
        size: Optional[int],
    ) -> None:
        self.input_dir: Path = input_directory
        self.bg_dir: Path = backgrounds_directory
        self.output_dir: Path = output_directory
        self.bg_color_range: list = []
        self.size: int = size

    def __change_bg__(self, video_path: Path, bg_path: Path) -> None | np.ndarray:
        if not video_path.is_file():
            sys.exit()

        if not bg_path.is_file():
            sys.exit()

        video = cv2.VideoCapture(str(video_path))
        background = cv2.VideoCapture(str(bg_path))

        ret, frame = video.read()
        if not ret:
            sys.exit()

        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        kernel = np.ones((5, 5), np.uint8)
        frames_list = []

        while video.isOpened():
            ret, frame = video.read()
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

        video.release()
        cv2.destroyAllWindows()

        return np.array(frames_list)
