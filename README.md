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
1. To download the source datasets used for this project, run
    ```shell
    python -m download_datasets [-o <output_folder>] [-c <config_file>]
    ```
    > :warning: `gdown` may refuse to download some parts of the dataset automatically.   
    > You will be prompted in the console with instructions to complete the download if it cannot finish on its own.   
    > If that's the case, simply follow the instructions in the console. 

2. You can download the images for backgrounds on [kaggle](https://www.kaggle.com/datasets/pavansanagapati/images-dataset?resource=download)
    - Make sure the `archive.zip` is in the path `data\archive.zip`
    - Run
        ```python
        python scripts/extract_pictures_archive.py
        ```
## Development
Setup pre-commit
```shell
pre-commit install
```

To run pre-commit, simply commit with git, or run
```shell
pre-commit run --all-files
```

## Dependencies
- Python >= 3.11
- ffmpeg >= 7.0 (for video processing)

## Installations

### 1. ffmpeg

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

## Commands

### 1. Prepocessing - resize videos

#### Availables options

- `-s, --size`: Target size in pixels (default: 224)
- `-d, --data-folder`: Path or glob pattern to the input data folder(s)
- `-o, --output-folder`: Path to the output folder (optional, defaults to input folder)
- `-v, --verbose`: Show detailed logs
- `-q, --quiet`: Show only warnings and errors

#### Usage

```shell
python -m preprocessing resize_videos -d <data_folder> [-o <output_folder>] [-s <size>] [-v] [-q]
```

#### Usage Examples

1. Resize videos in a specific directory to 224x224 (default size for ViViT and TimeSformer):
    ```bash
    python -m preprocessing resize_videos -d data/train_rgb_front_clips -s 224
    ```

2. Resize videos to 512x512:
    ```bash
    python -m preprocessing resize_videos -d data/train_rgb_front_clips -s 512
    ```
    > :information_source: This will create a `512x512` directory and force videos to be square (may distort aspect ratio)

3. Process multiple directories using glob pattern:
    ```bash
    python -m preprocessing resize_videos -d "data/*_clips" -s 224
    ```

4. Specify a different output folder:
    ```bash
    python -m preprocessing resize_videos -d "data/*_clips" -o data/output -s 224
    ```

5. Quiet mode (show only warnings and errors):
    ```bash
    python -m preprocessing resize_videos -d data/train_rgb_front_clips -q
    ```
--------------------------------------------------------------------------------------------

### Preprocessing - change background
Replace the green background in another video

#### Availables options

- `-d, --data-folder`: Path or glob pattern to the input data folder(s)
- `-o, --output-folder`: Path to the output folder (optional, defaults to input folder)
- `-v, --verbose`: Show detailed logs
- `-q, --quiet`: Show only warnings and errors

#### Usage
```python
python -m processing change_bg -d <data_folder> [-o <output_folder>] [-v] [-q]
```

>:information_source: Once the command has been run, the program executes on all `data_folder` files. This may take some time. Please be patient
--------------------------------------------------------------------------------------------

### Preprocessing - change background and resize
Apply workflow `change background -> resize video`

#### Availables options
The same as [resize video submodule](#1-prepocessing---resize-videos)

#### Usage
```python
python -m processing chg_n_res -d <data_folder> [-o <output_folder>] [-s <size>] [-v] [-q]
```

### Directory Structure

- Original videos must be placed in the `raw_videos/` subdirectory
- Resized videos will be saved in a subdirectory named according to the size (e.g., `224x224/`, `512x512/`)
- `change_bg` will create, if non-existent, and put the videos in `no_green_bg` folder of the data_folder
- `chg_n_res` will create, if non-existent, and put the videos in `{size}x{size}_no_green_bg` folder of the data_folder
- The folder containing the background's files must be a direct subfolder of data and named `backgrounds`. If not, please indicate the source of the backgrounds [here](preprocessing/bg_changer/config.py).

### Performance Analysis

The batch size can be adjusted in `preprocessing/resize_videos/video_resize_config.py` by modifying the `batch_size` parameter in [`PROCESSING_CONFIG`](preprocessing/config.py) based on your CPU and VRAM capabilities, but the current sequential implementation is recommended for reliability and simplicity.
