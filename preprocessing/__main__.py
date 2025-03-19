import glob
import sys
from pathlib import Path

import click
from loguru import logger

from preprocessing.config import PREPROCESSING_STARTING_MSG, PREPROCESSING_TYPES, PREPROCESSOR_TYPES

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from preprocessing.resize_videos.utils import process_directory
from preprocessing.resize_videos.video_resize_config import PROCESSING_CONFIG, SUPPORTED_EXTENSIONS


def setup_logger() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO" if PROCESSING_CONFIG["verbose"] else "WARNING",
    )


@click.command()
@click.argument("preproc_type", type=click.Choice(PREPROCESSING_TYPES))
@click.option("-d", "--data-folder", type=str, required=True, help="Path or glob pattern to the input data folder(s).")
@click.option(
    "-o",
    "--output-folder",
    type=click.Path(),
    help="Path to the output folder. Defaults to input folder if not specified.",
)
@click.option("-s", "--size", default=224, help="Target size for width and height (default: 224)")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed logs.")
@click.option("-q", "--quiet", is_flag=True, help="Show only warnings and errors.")
def main(preproc_type: str, data_folder: str, output_folder: str | None, size: int, verbose: bool, quiet: bool) -> None:
    """Resize videos to specified dimensions."""
    config = {
        "width": size,
        "height": size,
        "batch_size": PROCESSING_CONFIG["batch_size"],
        "verbose": verbose if verbose else not quiet,
    }

    setup_logger()
    logger.info(PREPROCESSING_STARTING_MSG[preproc_type].format(size=size))

    processor = PREPROCESSOR_TYPES[preproc_type]

    data_paths = glob.glob(data_folder, recursive=True)  # noqa: PTH207
    data_paths = [Path(p) for p in data_paths]

    if not data_paths:
        logger.error(f"No directories found matching pattern: {data_folder}")
        return

    logger.info(f"Found {len(data_paths)} directories to process")

    for data_path in data_paths:
        if not data_path.exists():
            logger.error(f"Directory not found: {data_path}")
            continue

        output_path = Path(output_folder) / data_path.name if output_folder else data_path

        logger.info(f"Processing directory: {data_path}")
        process_directory(processor, data_path, config, SUPPORTED_EXTENSIONS, output_path)

    logger.info("Process completed")


if __name__ == "__main__":
    main()
