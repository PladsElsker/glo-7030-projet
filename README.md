# Sign-terpreter Project
Session project for course [GLO-7030: Apprentissage par réseaux de neurones profonds](https://www.ulaval.ca/etudes/cours/glo-7030-apprentissage-par-reseaux-de-neurones-profonds)  
This project provides a model capable of translating a sign language video into text

## Requirements installation & venv activation
To install project's requirements, follow steps below:

1. Create a virtual environment to isolate the project and its dependencies
    ```shell
    python -m venv .env
    ```

2. Activate the new environment
    1. Linux and Mac
        ```shell
        source .env/bin/activate
        ```
    2. Windows
        ```shell
        .\.env\Scripts\activate
        ```

3. Install requirements
    ```shell
    python -m pip install -r requirements.txt
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

## Development
Setup pre-commit
```shell
pre-commit install
```

To run pre-commit, simply commit with git, or run
```shell
pre-commit run --all-files
```

## Prerequisites

- Python 3.11+
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

### Available Options

- `-s, --size`: Target size in pixels (default: 224)
- `-d, --data-folder`: Path or glob pattern to the input data folder(s)
- `-o, --output-folder`: Path to the output folder (optional, defaults to input folder)
- `-v, --verbose`: Show detailed logs
- `-q, --quiet`: Show only warnings and errors

### Usage

```shell
python -m preprocessing.resize_videos -d <data_folder> [-o <output_folder>] [-s <size>] [-v] [-q]
```

### Usage Examples

1. Resize videos in a specific directory to 224x224 (default size for ViViT and TimeSformer):
```bash
python -m preprocessing.resize_videos -d data/train_rgb_front_clips -s 224
```

2. Resize videos to 512x512:
```bash
python -m preprocessing.resize_videos -d data/train_rgb_front_clips -s 512
```
> :information_source: This will create a `512x512` directory and force videos to be square (may distort aspect ratio)

3. Process multiple directories using glob pattern:
```bash
python -m preprocessing.resize_videos -d "data/*_clips" -s 224
```

4. Specify a different output folder:
```bash
python -m preprocessing.resize_videos -d "data/*_clips" -o data/output -s 224
```

5. Quiet mode (show only warnings and errors):
```bash
python -m preprocessing.resize_videos -d data/train_rgb_front_clips -q
```

### Directory Structure

- Original videos must be placed in the `raw_videos/` subdirectory
- Resized videos will be saved in a subdirectory named according to the size (e.g., `224x224/`, `512x512/`)

### Performance Analysis

The batch size can be adjusted in `preprocessing/resize_videos/video_resize_config.py` by modifying the `batch_size` parameter in `PROCESSING_CONFIG` based on your CPU and VRAM capabilities, but the current sequential implementation is recommended for reliability and simplicity.

