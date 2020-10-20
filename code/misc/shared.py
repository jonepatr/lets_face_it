from pathlib import Path
import toml
import numpy as np
from collections import namedtuple, abc


def update(d, u):
    for k, v in u.items():
        if isinstance(v, abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


base_path = Path(__file__).resolve().parents[1]
config_path = base_path / "config.toml"

CONFIG = toml.load(config_path.open())

config_local_path = base_path / "config.local.toml"
if config_local_path.exists():
    CONFIG = update(CONFIG, toml.load(config_local_path.open()))

RANDOM_SEED = CONFIG["project"]["random_seed"]
DATA_DIR = Path(CONFIG["project"]["data_dir"])
BASE_DIR = Path(CONFIG["project"]["base_dir"])

flame_config = namedtuple(
    "Config",
    [
        "flame_model_path",
        "static_landmark_embedding_path",
        "dynamic_landmark_embedding_path",
        "batch_size",
        "use_face_contour",
        "shape_params",
        "expression_params",
        "use_3D_translation",
    ],
)
