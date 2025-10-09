import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_dataset():
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Define dataset and download path
    dataset = 'neerajx05/brain-tumor-dataset'
    download_path = 'data/brain_tumor'

    # Download and unzip dataset
    api.dataset_download_files(dataset, path=download_path, unzip=True)
    print(f"Dataset downloaded and extracted to {download_path}")

if __name__ == "__main__":
    download_dataset()
