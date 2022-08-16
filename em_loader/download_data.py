from em_loader.path import em_loader_root
import requests


ZIP_FILE_ID = '1oEP4hpSlyAy6vfwvLATf8ptkQCLLw-0g'
def download(base_dir=None):
    base_dir = base_dir or em_loader_root
    download_file_from_google_drive
    out_zip_path = f"{base_dir}/data/test.zip"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--base_dir", "-b", default="./")

    args = parser.parse_args()
    if not args.version or not args.base_dir:
        parser.print_help()
        quit()

    download(base_dir=args.base_dir)
