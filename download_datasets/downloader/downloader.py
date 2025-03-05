from pathlib import Path

from ..postprocess import POSTPROCESS_MAP
from ..config.attributes import SAVE_NAME_ATTRIBUTE, POSTPROCESS_ATTRIBUTE
from ..config.exceptions import ConfigError, PostprocessConfigError


class Downloader:
    def __init__(self, config: dict, name: str) -> None:
        try:
            self.output_file = config[SAVE_NAME_ATTRIBUTE]
        except KeyError:
            raise ConfigError(SAVE_NAME_ATTRIBUTE, name)

        self.postprocessors = []
        for key in config.get(POSTPROCESS_ATTRIBUTE, []):
            try:
                Postprocessor = POSTPROCESS_MAP[key]
                self.postprocessors.append(Postprocessor())
            except KeyError:
                raise PostprocessConfigError(key, name)

    def apply(self, directory: Path) -> Path:
        raise NotImplementedError()

    def postprocess(self, path: Path) -> None:
        for postprocessor in self.postprocessors:
            postprocessor.apply(path / self.output_file)
