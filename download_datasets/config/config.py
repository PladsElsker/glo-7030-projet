import os
from pathlib import Path

from .attributes import DOWNLOADER_TYPE_ATTRIBUTE
from .exceptions import ConfigError, DownloaderConfigError
from ..downloader import DOWNLOADER_MAP


class DatasetConfig:
    def __init__(self, json: dict) -> None:
        self.downloaders = {}
        for k, v in json.items():
            try:
                downloader_type = v[DOWNLOADER_TYPE_ATTRIBUTE]
            except KeyError:
                raise ConfigError(DOWNLOADER_TYPE_ATTRIBUTE, k)

            try:
                self.downloaders[k] = DOWNLOADER_MAP[downloader_type](v, k)
            except KeyError:
                raise DownloaderConfigError(downloader_type, k)

    def download(self, save_to: Path) -> None:
        if not os.path.exists(save_to):
            os.mkdir(save_to)

        for _, downloader in self.downloaders.items():
            downloader.download(save_to)
            downloader.postprocess(save_to)
