from download_datasets.downloader.common_downloader import Downloader
from download_datasets.downloader.registry import DOWNLOADER_MAP


class DownloaderFactory:
    def __init__(self, config: dict, name: str) -> None:
        self.config = config
        self.name = name

    def create(self, source_type: str) -> Downloader:
        downloader_cls = DOWNLOADER_MAP.get(source_type)
        if not downloader_cls:
            raise ValueError
        return downloader_cls(self.config, self.name)
