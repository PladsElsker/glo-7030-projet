from pathlib import Path


def without_suffix(path: Path) -> Path:
    while path.suffix:
        path = path.with_suffix("")

    return path
