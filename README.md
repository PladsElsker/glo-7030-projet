# Sign-terpreter
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
