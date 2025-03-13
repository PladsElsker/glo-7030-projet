# Sign-terpreter Project
Session project for course [GLO-7030: Apprentissage par réseaux de neurones profonds](https://www.ulaval.ca/etudes/cours/glo-7030-apprentissage-par-reseaux-de-neurones-profonds)  
This project provides a model capable of translating a sign language video into text

## Requirements installation & venv activation
To install project's requirements, follow steps below:

1. Create a virtual environment to isolate the project and its dependencies
    ```shell
    python -m venv .env
    ```

2. Install requirements
    ```shell
    python -m pip install -r requirements.txt
    ```

3. Activate the new environment
    1. Linux and Mac
        ```shell
        source .env/bin/activate
        ```
    2. Windows
        ```shell
        .\.env\Scripts\activate
        ```

> :information_source: All these commands are to be executed in the project root folder.

## Datasets
To download the source datasets used for this project, run
```shell
python -m download_datasets [-o <output_folder>] [-c <config_file>]
```
> :warning: `gdown` may refuse to download some parts of the dataset automatically.   
> You will be prompted in the console with instructions to complete the download if it cannot finish on its own.   
> If that's the case, simply follow the instructions in the console. 

## Prerequisites

- Python 3.8+
- ffmpeg (for video processing)

### Installing ffmpeg

1. **macOS** (using Homebrew):
    ```bash
    brew install ffmpeg
    ```

2. **Ubuntu/Debian**:
    ```bash
    sudo apt update
    sudo apt install ffmpeg
    ```

3. **Windows**:
    - Download the latest release from [ffmpeg official website](https://ffmpeg.org/download.html)
    - Extract the archive
    - Add the `bin` folder to your system's PATH

## Video Resizing

The `resize_videos.py` script allows you to resize videos while preserving the folder structure.

### Available Options

- `--size`: Target size in pixels (default: 224)
- `--data`: Specific directory to process
- `--all`: Process all clip directories
- `--overwrite`: Overwrite existing files
- `--not-square`: Maintain original aspect ratio (default: force square)

### Usage Examples

1. Resize videos in a specific directory to 224x224 (default size for ViViT and TimeSformer):
```bash
python -m preprocessing.scripts.resize_videos --size 512 --data data/train_rgb_front_clips
```

2. Resize while preserving aspect ratio:
```bash
python -m preprocessing.scripts.resize_videos --size 512 --data data/train_rgb_front_clips --not-square
```

3. Process all clip directories:
```bash
python -m preprocessing.scripts.resize_videos --size 224 --all
```

4. Overwrite existing files:
```bash
python -m preprocessing.scripts.resize_videos --size 224 --data data/train_rgb_front_clips --overwrite
```

### Directory Structure

- Original videos must be placed in the `raw_videos/` subdirectory
- Resized videos will be saved in a subdirectory named according to the size (e.g., `224x224/`, `512x512/`)