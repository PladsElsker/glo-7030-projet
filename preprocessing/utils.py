from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from preprocessing.config import DATASET_SPLITS, DATASET_VIEWS, PATHS
from preprocessing.video_processor import VideoProcessor


def create_output_dirs(output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)


def get_output_dir(size: int, preproc_typename: str) -> str:
    return PATHS[f"{preproc_typename}_output_dir_template"].format(size=size)


def get_clip_directories() -> list[str]:
    return [f"{split}_{view}_clips" for split in DATASET_SPLITS for view in DATASET_VIEWS]


def process_video_file(
    processor: VideoProcessor,
    input_path: Path | str,
    output_path: Path | str,
    config: dict,
) -> bool:
    input_path, output_path = Path(input_path), Path(output_path)

    if not input_path.exists():
        logger.warning(f"File not found: {input_path.name}")
        return False

    if output_path.exists() and not config.get("overwrite", False):
        return True

    if config.get("verbose", True):
        logger.info(f"Processing: {input_path.name}")

    success = processor.process(input_path, output_path, config)
    if success:
        logger.success(f"Successfully: {input_path.name}")
    else:
        logger.error(f"Failed: {input_path.name}")

    return success


def process_directory(
    processor: VideoProcessor,
    clip_dir: Path | str,
    config: dict,
    extensions: Sequence[str],
    output_path: Path | str,
) -> None:
    raw_videos_path = Path(clip_dir) / PATHS["raw_videos_dir"]
    output_path = Path(output_path) / get_output_dir(config["width"], type(processor).__name__)

    if not raw_videos_path.exists():
        logger.error(f"Directory not found: {raw_videos_path.name}")
        return

    create_output_dirs(output_path)
    video_files = [f for ext in extensions for f in raw_videos_path.glob(f"*{ext}")]

    if not video_files:
        logger.warning(f"No videos found in {raw_videos_path.name}")
        return

    with ThreadPoolExecutor(max_workers=config.get("batch_size", 1)) as executor:
        futures = [executor.submit(process_video_file, processor, video_file, output_path / video_file.name, config) for video_file in video_files]
        for future in as_completed(futures):
            future.result()

    logger.success(f"Successfully processed {Path(clip_dir).name}")


def process_all_datasets(
    processor: VideoProcessor,
    config: dict,
    extensions: Sequence[str],
) -> None:
    data_root = Path(PATHS["data_root"])

    for dataset in get_clip_directories():
        dataset_path = data_root / dataset
        if not dataset_path.exists():
            logger.warning(f"Dataset not found: {dataset}")
            continue

        logger.info(f"Processing: {dataset}")
        process_directory(processor, dataset_path, config, extensions, dataset_path)
        logger.success(f"Successfully processed dataset: {dataset}")
