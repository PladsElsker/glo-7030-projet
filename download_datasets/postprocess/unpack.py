import shutil
from pathlib import Path

from loguru import logger

from download_datasets.path_helpers import without_suffix

from .postprocess import Postprocessor


class Unpack(Postprocessor):
    def apply(self, source: Path) -> None:
        if not source.exists():
            logger.warning(f'Skipping unpack "{source}" as the file is already unpacked.')
            return

        target_directory = without_suffix(source)
        target_directory.mkdir(exist_ok=True, parents=True)
        logger.info(f'Unpacking "{source}" to "{target_directory}"...')
        shutil.unpack_archive(source, target_directory)
        logger.success(f'Unpacked "{source}".')
