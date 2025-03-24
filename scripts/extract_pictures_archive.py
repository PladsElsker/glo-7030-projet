# noqa: INP001
import shutil
import zipfile
from pathlib import Path

from loguru import logger
from tqdm import tqdm

ARCHIVE_FILE_PATH = Path(r"data/archive.zip")
BACKGROUNDS_PICTURES_FOLDER = Path(r"data/backgrounds/pictures")

with zipfile.ZipFile(str(ARCHIVE_FILE_PATH), "r") as zip_ref:
    unzipped_path = ARCHIVE_FILE_PATH.parent / "archive"
    members = zip_ref.infolist()
    logger.info("Extracting archive...")
    for member in tqdm(members, desc="Unzipping", unit="file"):
        zip_ref.extract(member, str(unzipped_path))

BACKGROUNDS_PICTURES_FOLDER.mkdir(parents=True, exist_ok=True)

data_path = Path(unzipped_path)
files_to_move = [f for f in data_path.rglob("*") if f.is_file()]

logger.info("Moving files to backgrounds/pictures...")
for file in tqdm(files_to_move, desc="Moving", unit="file"):
    destination = BACKGROUNDS_PICTURES_FOLDER / file.name
    shutil.move(str(file), str(destination))

shutil.rmtree(unzipped_path)
logger.success("Extraction, file move, and cleanup completed.")
