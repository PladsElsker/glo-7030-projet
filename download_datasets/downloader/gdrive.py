from pathlib import Path

import gdown
from gdown.exceptions import FileURLRetrievalError
from loguru import logger

from download_datasets.config.exceptions import ConfigError
from download_datasets.downloader.common_downloader import Downloader
from download_datasets.path_helpers import without_suffix

GDRIVE_ID_ATTRIBUTE = "Id"
GDRIVE_URL_FORMAT = "https://drive.google.com/uc?id={file_id}"


class GoogleDriveDownloader(Downloader):
    def __init__(self, config: dict, name: str) -> None:
        super().__init__(config, name)
        try:
            self.file_id = config[GDRIVE_ID_ATTRIBUTE]
        except KeyError as e:
            raise ConfigError(GDRIVE_ID_ATTRIBUTE, name) from e

    def download(self, directory: Path) -> Path:
        target = directory / self.output_file
        url = GDRIVE_URL_FORMAT.format(file_id=self.file_id)
        target_directory = without_suffix(target)

        if target.exists() or target_directory.exists():
            logger.success(f'Dataset "{target}" is already downloaded.')
            return target

        try:
            logger.info(f'Downloading "{target}"...')
            gdown.download(url, str(target), quiet=False)
        except FileURLRetrievalError:
            logger.error(f'Unable to download "{target}".')
            logger.info(f"Download the file manually at {url}.")
            logger.info(f'Once downloaded, move it under "{target.parents[0].resolve()}".')
            while not target.exists():
                input(f'File "{target}" not found. Press "Enter" when the file has been added: ')

        logger.success(f'Downloaded "{target}".')
        return target
