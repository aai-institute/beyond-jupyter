"""
This script downloads the spotify 1 million dataset from
Kaggle (https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset)
and unzips it to the subfolder 'data' on the top level.
The script leverages the Kaggle SDK for authenticating with the Kaggle API and downloading the dataset.
Make sure that the 'kaggle.json' file with the Kaggle API credentials is properly set up in the system.
You can follow the instructions at https://www.kaggle.com/docs/api.

The main function `load_via_kaggle` performs the authentication and download steps.

This script can be run directly, and upon execution, it will download and unzip the specified dataset to the
destination path.

Dependencies:
    - Kaggle SDK
    - os module

Usage:
    python load_data.py

"""
import os
from kaggle import KaggleApi

script_folder = os.path.dirname(os.path.abspath(__file__))
destination_path = os.path.join(script_folder, "data")
dataset_name = 'amitanshjoshi/spotify-1million-tracks'


def load_via_kaggle():
    # Authenticating with the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Downloading the dataset
    api.dataset_download_files(dataset_name, path=destination_path, unzip=True)


if __name__ == "__main__":
    load_via_kaggle()
