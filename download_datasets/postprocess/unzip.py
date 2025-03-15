import shutil
import zipfile
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from .postprocess import Postprocessor


class Unzip(Postprocessor):
    def apply(self, target: Path) -> None:
        if not target.exists():
            logger.warning(f'Skipping unzip "{target}" as the file is already unzipped.')
            return

        target.with_suffix("").mkdir(parents=True, exist_ok=True)

        self._extract_zip(str(target), str(target.with_suffix("")))

        logger.success(f'Unzipped "{target}".')

    def _extract_zip(self, zip_path: str, extract_to: str) -> None:
        logger.info(f'Extracting "{zip_path}"...')
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = [m for m in zip_ref.infolist() if not m.is_dir()]
            for member in tqdm(members, desc=f'Extracting "{zip_path}"'):
                try:
                    extracted_path = Path(extract_to) / member.filename
                    extracted_path.parent.mkdir(exist_ok=True, parents=True)
                    with zip_ref.open(member) as source, Path.open(extracted_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                except (zipfile.BadZipFile, OSError):
                    logger.error(f'Error unzipping "{member.filename}".')
