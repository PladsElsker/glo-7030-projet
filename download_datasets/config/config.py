import os
from pathlib import Path

from .attributes import DOWNLOADER_TYPE_ATTRIBUTE
from .exceptions import ConfigError, DownloaderConfigError
from ..downloader import DOWNLOADER_MAP


class DatasetConfig:
    def __init__(self, json: dict) -> None:
        self.datasets = {}
        for k, v in json.items():
            try:
                downloader_attribute = v[DOWNLOADER_TYPE_ATTRIBUTE]
            except KeyError:
                raise ConfigError(DOWNLOADER_TYPE_ATTRIBUTE, k)
            
            try:
                self.datasets[k] = DOWNLOADER_MAP[downloader_attribute](v, k)
            except KeyError:
                raise DownloaderConfigError(downloader_attribute, k)

    def download(self, save_to: Path) -> None:
        if not os.path.exists(save_to):
            os.mkdir(save_to)

        for _, downloader in self.datasets.items():
            downloader.apply(save_to)
            downloader.postprocess(save_to)
