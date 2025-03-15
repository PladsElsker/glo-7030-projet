import shutil
import tarfile
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from .postprocess import Postprocessor


class Untar(Postprocessor):
    def apply(self, target: Path) -> None:
        if not target.exists():
            logger.warning(f'Skipping untar "{target}" as the file is already untarred.')
            return

        target.with_suffix("").mkdir(exist_ok=True, parents=True)
        self._extract_tar(str(target), str(target.with_suffix("")))
        logger.success(f'Untarred "{target}".')

    def _extract_tar(self, tar_path: str, extract_to: Path) -> None:
        logger.info(f'Extracting "{tar_path}"...')
        with tarfile.open(tar_path, "r") as tar_ref:
            members = tar_ref.getmembers()
            for member in tqdm(members, desc=f'Extracting "{tar_path}"'):
                try:
                    extracted_path = extract_to / member.filename
                    extracted_path.parent.mkdir(exist_ok=True, parents=True)
                    with tar_ref.open(member) as source, Path.open(extracted_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                except (tarfile.TarError, OSError):
                    logger.error(f'Error untarring "{member.filename}".')
