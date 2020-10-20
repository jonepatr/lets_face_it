from psbody.mesh import Mesh
import os
from tqdm import tqdm
import sys
import skimage.io as io
import cv2
import pickle
import numpy as np
from glob import glob
import scipy.signal
from voca.utils.inference import inference2
from TF_FLAME.sample_FLAME_patrik import sample_FLAME, sample_FLAME2
from TF_FLAME.fit_2D_landmarks_patrik import fit_lmk2d
from TF_FLAME.fit_3D_mesh_patrik import fit_3D_mesh_v2, MeshFitter
from TF_FLAME.utils.landmarks import load_embedding
import RingNet.demo
import tensorflow as tf
from ray.util import ActorPool
import ray


ray_is_init = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def mesh_fitting(
    imgs,
    dest_path,
    ringnet_dir,
    landmark_dir,
    flame_lmk_path="/workspace/TF_FLAME/data/flame_static_embedding.pkl",
    texture_mapping="/workspace/TF_FLAME/data/texture_data.npy",
    template_model_fname="/models/flame_model/ch_models/generic_model.pkl",
    template_fname="./data/template.ply",
):
    lmk_face_idx, lmk_b_coords = load_embedding(flame_lmk_path)

    for file_ in tqdm(imgs, desc="mesh_fitting"):
        basename = os.path.basename(file_).rsplit(".", 1)[0]
        dest_path_sub_dir = os.path.join(dest_path, basename)

        if os.path.exists(dest_path_sub_dir):
            continue

        ringnet_sub_dir = os.path.join(ringnet_dir, basename)
        landmark_file = os.path.join(landmark_dir, basename + ".npy")

        neutral_mesh = os.path.join(ringnet_sub_dir, "neutral_mesh.ply")
        flame_parameters = pickle.load(
            open(os.path.join(ringnet_sub_dir, "flame_parameters.pkl"), "rb")
        )

        neutral_mesh = Mesh(filename=os.path.join(ringnet_sub_dir, "neutral_mesh.ply"))

        landmarks = np.load(landmark_file)

        target_img = cv2.imread(file_)
        result_mesh, _, out_flame_params = fit_lmk2d(
            target_img,
            landmarks,
            template_model_fname,
            neutral_mesh,
            lmk_face_idx,
            lmk_b_coords,
            flame_parameters,
        )

        os.makedirs(dest_path_sub_dir, exist_ok=True)

        result_mesh.write_obj(os.path.join(dest_path_sub_dir, basename + ".obj"))

        pickle.dump(
            out_flame_params,
            open(os.path.join(dest_path_sub_dir, "flame_parameters.pkl"), "wb"),
        )

        tf.reset_default_graph()
        tf.keras.backend.clear_session()


def extract_ringnet(
    imgs,
    dest_path,
    flame_sample_model="/models/flame_model/FLAME_sample.ply",
    template_model_fname="/models/flame_model/ch_models/generic_model.pkl",
):

    for file_ in tqdm(imgs, desc="extract_ringnet"):
        basename = os.path.basename(file_).rsplit(".", 1)[0]
        dest_path_sub_dir = os.path.join(dest_path, basename)

        if os.path.exists(dest_path_sub_dir):
            continue

        mesh, neutral_mesh, flame_parameters, input_img, proc_param = demo.main_2(
            file_, flame_sample_model, template_model_fname,
        )

        os.makedirs(dest_path_sub_dir, exist_ok=True)
        pickle.dump(mesh, open(os.path.join(dest_path_sub_dir, "mesh.pkl"), "wb"))
        mesh.write_obj(os.path.join(dest_path_sub_dir, "mesh.obj"))

        pickle.dump(
            neutral_mesh,
            open(os.path.join(dest_path_sub_dir, "neutral_mesh.pkl"), "wb"),
        )
        neutral_mesh.write_ply(os.path.join(dest_path_sub_dir, "neutral_mesh.ply"))

        pickle.dump(
            flame_parameters,
            open(os.path.join(dest_path_sub_dir, "flame_parameters.pkl"), "wb"),
        )

        pickle.dump(
            proc_param, open(os.path.join(dest_path_sub_dir, "proc_param.pkl"), "wb")
        )

        io.imsave(os.path.join(dest_path_sub_dir, "used_img.jpg"), input_img)

        tf.reset_default_graph()
        tf.keras.backend.clear_session()


def get_flame_parameters_for_objs(
    voca_objs, dest_path, model_fname="/models/flame_model/ch_models/generic_model.pkl",
):
    global ray_is_init
    if not ray_is_init:
        ray.init(num_gpus=2)
        ray_is_init = True

    MeshFitterActor = ray.remote(MeshFitter).options(num_gpus=0.01, num_cpus=1)

    dest_path.mkdir(parents=True, exist_ok=True)

    files = [x for x in voca_objs if not (dest_path / x.name).exists()]

    if not files:
        return [dest_path / x.name for x in voca_objs]

    cpu_count = int(ray.available_resources()["CPU"]) - 2

    actors = []
    for i in range(cpu_count):
        actors.append(MeshFitterActor.remote(model_fname))

    pool = ActorPool(actors)

    def run(a, file_):
        vertices = np.load(file_, allow_pickle=True)
        return a.fit.remote(vertices, dest_path / file_.name)

    dest_paths = []
    for dest_file_path, flame_params in tqdm(
        pool.map_unordered(lambda a, file_: run(a, file_), voca_objs),
        total=len(voca_objs),
    ):
        np.save(dest_file_path, flame_params)
        dest_paths.append(dest_file_path)

    return sorted(dest_paths)





def smooth_faces(obj_files, dest_path):
    faces = []

    for file_path in tqdm(obj_files, desc="smooth_faces: preparing"):
        mesh = Mesh(filename=file_path)
        faces.append(mesh.v)

    smooth_faces = scipy.signal.savgol_filter(
        np.stack(faces), 9, 3, mode="nearest", axis=0
    )
    os.makedirs(dest_path, exist_ok=True)
    for i, face in enumerate(tqdm(smooth_faces, desc="smooth_faces: saving")):
        dest_file_path = os.path.join(dest_path, os.path.basename(obj_files[i]))
        if os.path.exists(dest_file_path):
            continue
        Mesh(v=face, f=mesh.f).write_obj(dest_file_path)


def merge_face_with_voca(
    smooth_flame_paths,
    voca_flame_paths,
    ringnet_dir,
    dest_path,
    model_fname="/models/flame_model/ch_models/generic_model.pkl",
):

    os.makedirs(dest_path, exist_ok=True)

    for smooth_flame_path, voca_flame_path in tqdm(
        zip(smooth_flame_paths, voca_flame_paths),
        total=len(smooth_flame_paths),
        desc="merge_face_with_voca",
    ):
        base_name = os.path.basename(smooth_flame_path).rsplit(".", 1)[0]
        dest_file_path = os.path.join(dest_path, base_name + ".obj")
        if os.path.exists(dest_file_path):
            continue

        template_fname = os.path.join(ringnet_dir, base_name, "neutral_mesh.ply")
        voca_flame_parameters = pickle.load(open(voca_flame_path, "rb"))
        smooth_flame_parameters = pickle.load(open(smooth_flame_path, "rb"))
        mesh = sample_FLAME(
            template_fname, model_fname, voca_flame_parameters, smooth_flame_parameters
        )
        mesh.write_obj(dest_file_path)
        tf.reset_default_graph()
        tf.keras.backend.clear_session()


def merge_face_with_voca_with_neck_rot(
    smooth_flame_paths,
    voca_flame_paths,
    ringnet_dir,
    dest_path,
    model_fname="/models/flame_model/ch_models/generic_model.pkl",
):

    os.makedirs(dest_path, exist_ok=True)

    rots = []
    for smooth_flame_path in tqdm(
        smooth_flame_paths, desc="merge_face_with_voca_with_neck_rot"
    ):
        smooth_flame_parameters = pickle.load(open(smooth_flame_path, "rb"))
        rots.append(smooth_flame_parameters["rot"])
    avg_rot = np.vstack(rots).mean(axis=0)

    avg_rot[1:] = 0
    for smooth_flame_path, voca_flame_path in tqdm(
        zip(smooth_flame_paths, voca_flame_paths),
        total=len(smooth_flame_paths),
        desc="merge_face_with_voca_with_neck_rot",
    ):
        base_name = os.path.basename(smooth_flame_path).rsplit(".", 1)[0]
        dest_file_path = os.path.join(dest_path, base_name + ".obj")
        if os.path.exists(dest_file_path):
            continue

        template_fname = os.path.join(ringnet_dir, base_name, "neutral_mesh.ply")

        voca_flame_parameters = pickle.load(open(voca_flame_path, "rb"))
        smooth_flame_parameters = pickle.load(open(smooth_flame_path, "rb"))
        smooth_flame_parameters["rot"] -= avg_rot

        mesh = sample_FLAME2(
            template_fname, model_fname, voca_flame_parameters, smooth_flame_parameters,
        )
        mesh.write_obj(dest_file_path)
        tf.reset_default_graph()
        tf.keras.backend.clear_session()
