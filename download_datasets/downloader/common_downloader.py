from pathlib import Path

from download_datasets.config.attributes import (
    POSTPROCESS_ATTRIBUTE,
    SAVE_NAME_ATTRIBUTE,
)
from download_datasets.config.exceptions import ConfigError, PostprocessConfigError
from download_datasets.postprocess import POSTPROCESS_MAP


class Downloader:
    def __init__(self, config: dict, name: str) -> None:
        try:
            self.output_file = config[SAVE_NAME_ATTRIBUTE]
        except KeyError as e:
            raise ConfigError(SAVE_NAME_ATTRIBUTE, name) from e

        self.postprocessors = []
        for key in config.get(POSTPROCESS_ATTRIBUTE, []):
            try:
                postprocessor = POSTPROCESS_MAP[key]
                self.postprocessors.append(postprocessor())
            except KeyError as e:
                raise PostprocessConfigError(key, name) from e

    def download(self, directory: Path) -> Path:
        raise NotImplementedError

    def postprocess(self, path: Path) -> None:
        for postprocessor in self.postprocessors:
            postprocessor.apply(path / self.output_file)
