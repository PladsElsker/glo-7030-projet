import json
from pathlib import Path

import click
from config import DatasetConfig


@click.command()
@click.option("--output-directory", "-o", default="data", type=click.Path(file_okay=False, dir_okay=True), help="Directory to save downloaded files")
@click.option("--config-file", "-c", default="download_datasets/config.json", type=click.Path(exists=True), help="Path to JSON config file")
def download_datasets(output_directory: str, config_file: str) -> None:
    with Path.open(config_file) as f:
        config_json = json.load(f)

    output_directory = Path(output_directory)

    config = DatasetConfig(config_json)
    config.download(save_to=output_directory)


if __name__ == "__main__":
    download_datasets()
