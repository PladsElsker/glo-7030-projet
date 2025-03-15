from typing import Final

PATHS: Final[dict[str, str]] = {
    "data_root": "data",
    "raw_videos_dir": "raw_videos",
    "output_dir_template": "{size}x{size}",
}

PROCESSING_CONFIG: Final[dict[str, bool | int]] = {
    "batch_size": 1,
    "overwrite": False,
    "verbose": True,
}

SUPPORTED_EXTENSIONS: Final[list[str]] = [".mp4", ".avi", ".mov", ".mkv"]

DATASET_SPLITS: Final[list[str]] = ["train", "val", "test"]
DATASET_VIEWS: Final[list[str]] = ["rgb_front", "rgb_side"]


def get_resize_config(size: int = 224) -> dict[str, int]:
    return {
        "width": size,
        "height": size,
    }


def get_clip_directories() -> list[str]:
    return [f"{split}_{view}_clips" for split in DATASET_SPLITS for view in DATASET_VIEWS]


def get_output_dir(size: int) -> str:
    return PATHS["output_dir_template"].format(size=size)
