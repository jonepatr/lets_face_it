import pickle
import re

import numpy as np
from tqdm import tqdm

from misc.shared import BASE_DIR, DATA_DIR
from misc.utils import get_gender
from RingNet.util.using_flame_parameters import make_prdicted_mesh_neutral_2

for dir_ in tqdm(list(DATA_DIR.glob("Sessions_50fps_ringnet/*/*"))):
    participant = re.search(r".*_(.+?)_FaceNear", dir_.name).group(1)
    gender = get_gender(dir_.parent.name, participant)

    shapes = []
    for npy_file in tqdm(list(dir_.glob("flame_params/*.npy"))):
        data = np.load(npy_file, allow_pickle=True, encoding="latin1").item()
        shapes.append(data["shape"])

    avg_shape = np.vstack(shapes).mean(axis=0)

    neutral_mesh = make_prdicted_mesh_neutral_2(
        {"shape": avg_shape},
        BASE_DIR / "models/flame_model/ch_models/" / f"{gender}_model.pkl",
    )

    pickle.dump(neutral_mesh, (dir_ / "neutral_mesh.pkl").open("wb"))
    neutral_mesh.write_ply(str(dir_ / "neutral_mesh.ply"))
