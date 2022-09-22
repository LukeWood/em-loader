import os
from os.path import expanduser

home = expanduser("~")
em_loader_root = f"{home}/data/em-loader"
em_loader_root = os.path.abspath(em_loader_root)

def ensure_exists(path):
    os.makedirs(path, exist_ok=True)


def get_base_dir():
    ensure_exists(em_loader_root)
    ensure_exists(f"em_loader_root/data")
    return em_loader_root
