import os
import json
import requests
import click
from tqdm import tqdm
from pathlib import Path
from urllib.parse import urlparse


SOURCE_DATASETS = 'Sources'


def download(output_directory: str, config_file: str, chunk_size: int):
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    with open(config_file, 'r') as f:
        config = json.load(f)

    source_dataset_urls = config[SOURCE_DATASETS]

    for url in source_dataset_urls:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            parsed_url = urlparse(url)
            filename = parsed_url.path.replace('/', '_')

            output_path = Path(output_directory) / filename

            with open(output_path, 'wb') as file, tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as progress:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
                        progress.update(len(chunk))


@click.command()
@click.option('--output-directory', '-o', default='data', type=click.Path(file_okay=False, dir_okay=True), help='Directory to save downloaded files')
@click.option('--config-file', '-c', default='data_loader/datasets.json', type=click.Path(exists=True), help='Path to JSON config file')
@click.option('--chunk-size', '-s', default=8192, type=int, help='Download chunk size in bytes')
def cli(output_directory, config_file, chunk_size):
    download(
        output_directory=output_directory, 
        config_file=config_file, 
        chunk_size=chunk_size
    )


if __name__ == '__main__':
    cli()
