import json
import random
import os
from pathlib import Path

import tensorflow as tf
from tqdm.auto import tqdm

from em_loader.path import em_loader_root
import em_loader.download_data as download_lib


def load_labels(path, data_dir, verbosity):
    pass

def load(
    split="train", data_dir=None, download=True, verbosity=1, with_info=True
):
    """the primary entrypoint to load the halochromic dataset."""
    pass
