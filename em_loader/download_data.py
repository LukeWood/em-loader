import argparse
import zipfile

import requests

from em_loader.path import em_loader_root

data_path = "http://noiselab.ucsd.edu/sig_images.zip"


def download(base_dir=None):
    base_dir = base_dir or em_loader_root
    response = requests.get(data_path)

    zip_file = f"{base_dir}/data/version-1.zip"
    target_dir = f"{base_dir}/data/version-1"

    open(zip_file, "wb").write(response.content)
    with zipfile.ZipFile(zip_file, "r") as zip:
        zip.extractall(target_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--base_dir", "-b", default="./")

    args = parser.parse_args()
    if not args.base_dir:
        parser.print_help()
        quit()

    download(base_dir=args.base_dir)
