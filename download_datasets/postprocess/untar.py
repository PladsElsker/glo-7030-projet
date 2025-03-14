import tarfile
import os
import shutil
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from .postprocess import Postprocessor


class Untar(Postprocessor):
    def apply(self, target: Path) -> None:
        if not os.path.exists(target):
            logger.warning(f'Skipping untar "{target}" as the file is already untarred.')
            return

        if not os.path.exists(target.with_suffix('')):
            os.mkdir(target.with_suffix(''))

        self._extract_tar(str(target), str(target.with_suffix('')))

        logger.success(f'Untarred "{target}".')

    def _extract_tar(self, tar_path: str, extract_to: Path) -> None:
        logger.info(f'Extracting "{tar_path}"...')
        with tarfile.open(tar_path, 'r') as tar_ref:
            members = tar_ref.getmembers()
            for member in tqdm(members, desc=f'Extracting "{tar_path}"'):
                try:
                    extracted_path = os.path.join(extract_to, member.filename)
                    os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
                    with tar_ref.open(member) as source, open(extracted_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                except (tarfile.TarError, OSError, IOError) as e:
                    logger.error(f'Error untarring "{member.filename}".')
