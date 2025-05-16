# ingestion.py
# Calling: Ingesting the Laptop Price Prediction dataset from Kaggle

import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Define dataset name and output directory
KAGGLE_DATASET = 'mrsimple07/laptoppriceprediction'
DOWNLOAD_DIR = 'data/raw'
ZIP_FILENAME = 'laptoppriceprediction.zip'

# Create the download directory if it does not exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Authenticate with the Kaggle API
api = KaggleApi()
api.authenticate()

# Download the dataset as a zip file
print("Downloading dataset...")
api.dataset_download_files(KAGGLE_DATASET, path=DOWNLOAD_DIR, unzip=False)

# Unzip the dataset
zip_path = os.path.join(DOWNLOAD_DIR, ZIP_FILENAME)
print("Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(DOWNLOAD_DIR)

# Remove the zip file after extraction
os.remove(zip_path)
print("Dataset is available in:", DOWNLOAD_DIR)