import shutil
from pathlib import Path

from loguru import logger
from tqdm import tqdm


class MoveFilesFromSubFoldersToFolder:
    def apply(self, target: Path) -> None:
        files_to_move = [f for f in target.rglob("*") if f.is_file()]
        logger.info("Moving files to backgrounds/pictures...")
        for f in tqdm(files_to_move, desc="Moving", unit="file"):
            destination = f.name
            shutil.move(str(f), str(destination))
        logger.success("Extraction, file move, and cleanup completed.")
