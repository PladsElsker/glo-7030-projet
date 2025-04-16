from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class VideoProcessor(ABC):

    @abstractmethod
    def process(self, input_path: Path, output_path: Path, config: dict[str, Any]) -> bool:
        pass
