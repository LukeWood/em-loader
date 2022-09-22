import os
from os.path import expanduser
import luketils

home = expanduser("~")
em_loader_root = f"{home}/data/em-loader"
em_loader_root = os.path.abspath(em_loader_root)

def ensure_exists(path):
    """ensure a nested directory exists."""
     os.makedirs(path, exist_ok=True)


def get_base_dir():
    ensure_exists(em_loader_root)
    return em_loader_root
