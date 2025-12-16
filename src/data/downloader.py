import os
import shutil
import kagglehub
from pathlib import Path

BASE_DIR = Path.cwd().parent.parent

def download_isic_data(target_dir=f"{BASE_DIR}\data\\raw"):
    print(f"Downloading dataset to {target_dir}...")
    try:
        path = kagglehub.dataset_download("salviohexia/isic-2019-skin-lesion-images-for-classification")
        os.makedirs(target_dir, exist_ok=True)
        shutil.copytree(path, target_dir, dirs_exist_ok=True)
        print("Dataset downloaded and copied to:", target_dir)
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_isic_data()