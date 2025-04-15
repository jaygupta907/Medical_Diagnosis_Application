import argparse
import logging
import os
import subprocess
import zipfile


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser(description='Training Arguments')
parser.add_argument('--download_url',type=str,default='https://www.kaggle.com/api/v1/datasets/download/prashant268/chest-xray-covid19-pneumonia')
parser.add_argument('--zip_path',type=str,default='chestxray.zip')
parser.add_argument('--dataset_path',type=str,default='datasets/chestxray')
args = parser.parse_args()
 


def download_and_extract_dataset(args):
    if not os.path.exists(args.dataset_path):
        logging.info("Downloading the Dataset ..........")
        subprocess.run(["curl","-L", args.download_url, "-o", args.zip_path], check=True)
        logging.info("Extracting the Dataset ..........")
        with zipfile.ZipFile(args.zip_path, "r") as zip_ref:
            zip_ref.extractall(path=args.dataset_path)
        os.remove(args.zip_path)
        logging.info("Dataset downloaded and extracted")
    else:
        logging.warning("Dataset already exists")

if __name__ == "__main__":
    download_and_extract_dataset(args)
