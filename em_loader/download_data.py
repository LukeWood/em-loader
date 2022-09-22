import argparse
import zipfile

import requests

from em_loader.path import get_base_dir

data_paths = {
    1: "http://noiselab.ucsd.edu/sig_images.zip",
    2: "http://noiselab.ucsd.edu/sig_images2.zip",
}


def download(version=2, base_dir=None):
    global data_paths
    base_dir = base_dir or get_base_dir()
    if not version in data_paths:
        raise ValueError(
            "Attempted to download an invalid version of the sig images. "
            f"Expected version to be in {data_paths.keys()}, but got version={version}"
        )
    data_path = data_paths[version]
    response = requests.get(data_path)

    zip_file = f"{base_dir}/data/version-{version}.zip"
    target_dir = f"{base_dir}/data/version-{version}"

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
