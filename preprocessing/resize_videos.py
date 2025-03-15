import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.utils import process_all_datasets, process_directory
from preprocessing.video_processor import FFmpegProcessor
from preprocessing.video_resize_config import PROCESSING_CONFIG, SUPPORTED_EXTENSIONS


def setup_logger() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO" if PROCESSING_CONFIG["verbose"] else "WARNING",
    )


@click.command()
@click.option("--size", default=224, help="Target size for width and height (default: 224)")
@click.option("--data", help="Specific directory to process")
@click.option("--process-all", is_flag=True, help="Process all clip directories")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files")
@click.option("--verbose/--quiet", default=True, help="Show detailed logs (default: True)")
def main(size: int, data: Optional[str], process_all: bool, overwrite: bool, verbose: bool) -> None:
    """Resize videos to specified dimensions."""
    config = {
        "width": size,
        "height": size,
        "batch_size": PROCESSING_CONFIG["batch_size"],
        "overwrite": overwrite,
        "verbose": verbose,
    }

    setup_logger()
    logger.info(f"Starting video resizing process with size {size}x{size}")

    processor = FFmpegProcessor()

    if data:
        data_path = Path(data)
        if not data_path.exists():
            logger.error(f"Directory not found: {data_path}")
            return
        logger.info(f"Processing specific directory: {data_path}")
        process_directory(processor, data_path, config, SUPPORTED_EXTENSIONS)
    elif process_all:
        logger.info("Processing all clip directories")
        process_all_datasets(processor, config, SUPPORTED_EXTENSIONS)
    else:
        logger.error("Please specify either --data <directory> or --process-all")
        return

    logger.info("Process completed")


if __name__ == "__main__":
    main()
