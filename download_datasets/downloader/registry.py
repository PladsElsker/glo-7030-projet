from .gdrive import GoogleDriveDownloader
from .kaggle import KaggleHubDownloader

DOWNLOADER_MAP = {"GoogleDrive": GoogleDriveDownloader, "KaggleHub": KaggleHubDownloader}
