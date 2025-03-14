import shutil
from pathlib import Path

from loguru import logger

from .postprocess import Postprocessor


class Delete(Postprocessor):
    def apply(self, target: Path) -> None:
        if Path.isfile(target):
            Path.unlink(target)
            logger.success(f'Deleted "{target}".')
        elif Path.isdir(target):
            shutil.rmtree(target)
            logger.success(f'Deleted "{target}".')
        else:
            logger.warning(f'Skipping delete "{target}" as the file does not exist.')
