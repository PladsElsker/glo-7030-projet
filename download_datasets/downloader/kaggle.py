import shutil
from pathlib import Path

import kagglehub as kglhub
from downloader import Downloader


class KaggleDownloader(Downloader):
    def __init__(self, config: dict, name: str) -> None:
        super().__init__(config, name)

    def __move_to_directory__(self, origin_path: Path, new_path: Path) -> None:
        shutil.move(str(origin_path), str(new_path))

    def download(self, directory: Path) -> Path:
        download_path = kglhub.dataset_download()
        self.__move_to_directory__(Path(download_path), directory)
