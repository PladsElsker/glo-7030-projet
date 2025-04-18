from .gdrive import GoogleDriveDownloader
from .wget import WGetDownloader

DOWNLOADER_MAP = {
    "GoogleDrive": GoogleDriveDownloader,
    "WGet": WGetDownloader,
}
