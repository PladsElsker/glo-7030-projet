import shutil
from pathlib import Path

import kagglehub as kgl
from kagglehub.exceptions import KaggleApiHTTPError
from loguru import logger
from tqdm import tqdm


class KaggleDownloader:
    def __init__(self) -> None:
        pass

    @staticmethod
    def __move__(origin: Path, destination: Path) -> Path | None:

        destination.mkdir(parents=True, exist_ok=True)

        logger.info(f"Moving files from {origin} to {destination}.All files will be placed in the root of {destination}")

        files_to_move = [f for f in origin.rglob("*") if f.is_file()]

        if not files_to_move:
            logger.warning(f"No files to move at {origin}")
            return None

        for file in tqdm(files_to_move, "Files moving", unit="file", colour="white"):
            file_destination = destination / file.name
            shutil.move(str(file), str(file_destination))

        logger.success(f"Files moved to {destination} successfully")

        return destination

    def download(self, destination: Path) -> Path:
        logger.info("Start downloading dataset A from Kaggle")
        dataset_downloaded_path: str | None = None

        try:
            dataset_downloaded_path = kgl.dataset_download("pavansanagapati/images-dataset")

        except (OSError, KaggleApiHTTPError, NotImplementedError) as e:
            logger.error(e)

        else:
            logger.success("Dataset Downloaded successfully")

        return self.__move__(Path(dataset_downloaded_path), destination)


if __name__ == "__main__":
    kgl_downloader = KaggleDownloader()
    destination = Path(r"D:\Cours\GLO-4030\SignTerpreter\data\backgrounds\pictures")
    kgl_downloader.download(destination)
