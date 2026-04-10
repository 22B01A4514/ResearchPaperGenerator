"""
Kaggle Dataset Downloader for arXiv Dataset
Downloads the complete arXiv dataset from Kaggle
Dataset: Cornell-University/arxiv (1.7M+ papers)
"""

import os
import subprocess
import sys
from dotenv import load_dotenv
load_dotenv()
class KaggleDownloader:
    def __init__(self, username, key, data_dir="data"):
        self.username = username
        self.key = key
        self.data_dir = data_dir
        self.dataset_name = "Cornell-University/arxiv"
        self.dataset_file = os.path.join(data_dir, "arxiv-metadata-oai-snapshot.json")

        os.makedirs(data_dir, exist_ok=True)

        #  Set environment variables
        os.environ["KAGGLE_USERNAME"] = self.username
        os.environ["KAGGLE_KEY"] = self.key

    def install_kaggle_api(self):
        """Install Kaggle API if not installed"""
        try:
            import kaggle
            print(" Kaggle API already installed")
            return True
        except ImportError:
            print("Installing Kaggle API...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
            print(" Kaggle API installed successfully")
            return True

    def download_dataset(self):
        """Download arXiv dataset from Kaggle"""

        # Check if already downloaded
        if os.path.exists(self.dataset_file):
            file_size = os.path.getsize(self.dataset_file) / (1024**3)
            print(f" Dataset already exists: {self.dataset_file}")
            print(f"  Size: {file_size:.2f} GB")
            return True

        print("\n Downloading arXiv dataset from Kaggle...")
        print(f"   Dataset: {self.dataset_name}")
        print("   Size: ~3.5 GB (this may take some time)")
        print(f"   Destination: {self.data_dir}/\n")

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            api.dataset_download_files(
                self.dataset_name,
                path=self.data_dir,
                unzip=True
            )

            print("\n Dataset downloaded successfully!")

            if os.path.exists(self.dataset_file):
                file_size = os.path.getsize(self.dataset_file) / (1024**3)
                print(f"File verified: {file_size:.2f} GB")
                return True
            else:
                print(" Download completed but file not found")
                return False

        except Exception as e:
            print(f"\n Error downloading dataset: {str(e)}")
            return False

    def get_dataset_info(self):
        """Get information about the dataset"""
        if not os.path.exists(self.dataset_file):
            return None

        total_papers = 0
        with open(self.dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                total_papers += 1
                if total_papers % 100000 == 0:
                    print(f"Counting papers: {total_papers:,}...")

        file_size = os.path.getsize(self.dataset_file) / (1024**3)

        return {
            'total_papers': total_papers,
            'file_size_gb': file_size,
            'file_path': self.dataset_file
        }


def setup_kaggle(username, key):
    downloader = KaggleDownloader(username, key)

    print("\n" + "="*70)
    print("ARXIV DATASET SETUP")
    print("="*70)

    # Step 1: Install API
    if not downloader.install_kaggle_api():
        return False

    # Step 2: Download dataset
    if downloader.download_dataset():
        info = downloader.get_dataset_info()
        if info:
            print(f"\n Dataset Information:")
            print(f"   Total Papers: {info['total_papers']:,}")
            print(f"   File Size: {info['file_size_gb']:.2f} GB")
            print(f"   File Path: {info['file_path']}")
        return True

    return False

if __name__ == "__main__":
    #  ENTER YOUR NEWLY GENERATED DETAILS HERE
    KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
    KAGGLE_KEY = os.getenv("KAGGLE_KEY")

    result = setup_kaggle(KAGGLE_USERNAME, KAGGLE_KEY)

    if result:
        print("\n Full dataset ready to use!")
    else:
        print("\n Setup failed")
