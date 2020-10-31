import contextlib
import os
import shutil
import tempfile
from collections import defaultdict, namedtuple
from pathlib import Path

import chumpy as ch
import h5py
import numpy as np
import skimage.io as io
from misc.shared import BASE_DIR, CONFIG, DATASET_DIR
from misc.utils import get_gender
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from psbody.mesh import Mesh
from RingNet.run_RingNet import RingNet_inference
from RingNet.smpl_webuser.serialization import load_model
from RingNet.smpl_webuser.verts import verts_decorated
from RingNet.util import image as img_util

Config = namedtuple(
    "Config",
    [
        "load_path",
        "batch_size",
        "img_size",
        "data_format",
        "pose_params",
        "shape_params",
        "expression_params",
    ],
)


def preprocess_image(img_path, img_size):
    """
    Taken (and modified) from the RingNet repo
    """
    img = io.imread(img_path)

    if np.max(img.shape[:2]) != img_size:
        scale = float(img_size) / np.max(img.shape[:2])
    else:
        scale = 1.0  # scaling_factor
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)

    # image center in (x,y)
    center = center[::-1]
    crop, proc_param = img_util.scale_and_crop(img, scale, center, img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.0) - 0.5)

    return crop, proc_param


def make_predicted_mesh_neutral(params, flame_model_path):
    """
    Taken (and modified) from the RingNet repo
    """
    pose = np.zeros(15)
    expression = np.zeros(100)
    shape = np.hstack((params["shape"], np.zeros(300 - params["shape"].shape[0])))
    flame_genral_model = load_model(flame_model_path)
    generated_neutral_mesh = verts_decorated(
        ch.array([0.0, 0.0, 0.0]),
        ch.array(pose),
        ch.array(flame_genral_model.r),
        flame_genral_model.J_regressor,
        ch.array(flame_genral_model.weights),
        flame_genral_model.kintree_table,
        flame_genral_model.bs_style,
        flame_genral_model.f,
        bs_type=flame_genral_model.bs_type,
        posedirs=ch.array(flame_genral_model.posedirs),
        betas=ch.array(
            np.hstack((shape, expression))
        ),  # betas=ch.array(np.concatenate((theta[0,75:85], np.zeros(390)))), #
        shapedirs=ch.array(flame_genral_model.shapedirs),
        want_Jtr=True,
    )
    return Mesh(v=generated_neutral_mesh.r, f=generated_neutral_mesh.f)


def extract_ringnet(fps):
    sess = tf.compat.v1.Session()

    config = Config(
        load_path=str(BASE_DIR / CONFIG["ringnet"]["model"]),
        batch_size=1,
        img_size=224,
        data_format="NHWC",
        pose_params=6,
        shape_params=100,
        expression_params=50,
    )
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        model = RingNet_inference(config, sess,)

    img_dirs = list(DATASET_DIR.glob(f"*/*/imgs_{fps}fps"))
    for img_dir in tqdm(img_dirs, desc="Extracting ringnet", leave=True):
        ringnet_file = img_dir.parent / f"ringnet_{fps}fps.h5"

        if ringnet_file.exists():
            continue

        data = defaultdict(list)
        for img in tqdm(list(sorted(img_dir.glob("*.jpg"))), leave=False):
            input_img, proc_param = preprocess_image(img, config.img_size)

            vertices, flame_parameters = model.predict(
                np.expand_dims(input_img, axis=0), get_parameters=True
            )

            data["proc_params/scale"].append(proc_param["scale"])
            data["proc_params/start_pt"].append(proc_param["start_pt"])
            data["proc_params/end_pt"].append(proc_param["end_pt"])
            data["proc_params/img_size"].append(proc_param["img_size"])
            data["vertices"].append(vertices[0])
            data["flame_params/cam"].append(flame_parameters[0][:3])
            data["flame_params/pose"].append(
                flame_parameters[0][3 : 3 + config.pose_params]
            )
            data["flame_params/shape"].append(
                flame_parameters[0][
                    3
                    + config.pose_params : 3
                    + config.pose_params
                    + config.shape_params
                ]
            )
            data["flame_params/expression"].append(
                flame_parameters[0][3 + config.pose_params + config.shape_params :]
            )

        with tempfile.TemporaryDirectory() as tmpd:
            tmpf = Path(tmpd) / ringnet_file.name
            with h5py.File(tmpf, "w") as f:
                for key, value in data.items():
                    arrays = np.stack(value)
                    if (arrays == arrays[0]).all():
                        arrays = arrays[0]

                    f.create_dataset(key, data=arrays)
            shutil.move(tmpf, ringnet_file)

    sess.close()


def extract_neutral_mesh(fps):
    files = list(DATASET_DIR.glob(f"*/*/ringnet_{fps}fps.h5"))
    for file in tqdm(files, desc="Extracting neutral mesh"):
        neutral_mesh_file = file.parent / "neutral_mesh.ply"
        if neutral_mesh_file.exists():
            continue

        gender = get_gender(file.parent.parent.name, file.parent.name)
        with h5py.File(file, "r") as f:
            avg_shape = f["flame_params/shape"][()].mean(axis=0)

        flame_model_path = BASE_DIR / CONFIG["flame"][f"model_path_{gender}"]
        neutral_mesh = make_predicted_mesh_neutral(
            {"shape": avg_shape}, flame_model_path
        )
        neutral_mesh.write_ply(str(neutral_mesh_file))
