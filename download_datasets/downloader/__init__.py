from .gdrive import GoogleDriveDownloader
from .kaggle import KaggleDownloader

DOWNLOADER_MAP = {"GoogleDrive": GoogleDriveDownloader, "Kaggle": KaggleDownloader}
