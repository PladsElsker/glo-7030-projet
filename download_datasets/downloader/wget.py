from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm

from download_datasets.config.exceptions import ConfigError
from download_datasets.path_helpers import without_suffix

from .downloader import Downloader

WGET_URL_ATTRIBUTE = "URL"
WGET_DOWNLOAD_CHUNK_SIZE = 8192


class WGetDownloader(Downloader):
    def __init__(self, config: dict, name: str) -> None:
        super().__init__(config, name)
        self.name = name
        try:
            self.url = config[WGET_URL_ATTRIBUTE]
        except KeyError as e:
            raise ConfigError(WGET_URL_ATTRIBUTE, name) from e

    def download(self, directory: Path) -> Path:
        target = directory / self.output_file

        target_directory = without_suffix(target)

        if target.exists() or target_directory.exists():
            logger.success(f'Dataset "{self.name}" is already downloaded.')
            return

        with requests.get(self.url, stream=True, timeout=3600) as r:
            r.raise_for_status()
            with Path.open(target, "wb") as f:
                total = int(r.headers.get("Content-Length", 0)) // WGET_DOWNLOAD_CHUNK_SIZE
                for chunk in tqdm(r.iter_content(chunk_size=WGET_DOWNLOAD_CHUNK_SIZE), desc=f"Download {self.name}", total=total):
                    f.write(chunk)
