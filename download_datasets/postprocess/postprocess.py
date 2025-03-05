from pathlib import Path


class Postprocessor:
    def apply(self, target: Path):
        raise NotImplementedError()
