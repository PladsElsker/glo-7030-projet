from pathlib import Path

from download_datasets.downloader.registry import DOWNLOADER_MAP

from .attributes import DOWNLOADER_TYPE_ATTRIBUTE
from .exceptions import ConfigError, DownloaderConfigError


class DatasetConfig:
    def __init__(self, json: dict) -> None:
        self.downloaders = {}
        for k, v in json.items():
            try:
                downloader_type = v[DOWNLOADER_TYPE_ATTRIBUTE]
            except KeyError as e:
                raise ConfigError(DOWNLOADER_TYPE_ATTRIBUTE, k) from e

            try:
                downloader = DOWNLOADER_MAP[downloader_type](v, k)
                self.downloaders[k] = downloader
            except KeyError as e:
                raise DownloaderConfigError(downloader_type, k) from e

    def download(self, save_to: Path) -> None:
        save_to.mkdir(parents=True, exist_ok=True)

        for _, downloader in self.downloaders.items():
            downloader.download(save_to)
            downloader.postprocess(save_to)
