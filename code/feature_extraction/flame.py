import os

import h5py
import numpy as np
import ray
from misc.shared import BASE_DIR, CONFIG, DATASET_DIR
from misc.utils import get_gender
from psbody.mesh import Mesh
from ray.util import ActorPool
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from TF_FLAME.utils.landmarks import load_embedding, tf_project_points


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt
from TF_FLAME.tf_smpl.batch_smpl import SMPL


@ray.remote(num_gpus=0.0625, num_cpus=0.5)
class FrameOptimizer(object):
    def __init__(self, neutral_mesh_faces, template_path):

        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()

        self.graph1 = tf.Graph()
        with self.graph1.as_default():
            weights = {
                "lmk": 1.0,
                "shape": 1e-3,
                "expr": 1e-3,
                "neck_pose": 100.0,
                "jaw_pose": 1e-3,
                "eyeballs_pose": 10.0,
            }

            self.template_mesh = tf.constant(neutral_mesh_faces)

            self.target_2d_lmks_x = tf.Variable(np.zeros((51, 1)))
            self.target_2d_lmks_y = tf.Variable(np.zeros((51, 1)))

            self.target_2d_lmks = tf.concat(
                [self.target_2d_lmks_x, 1024 - self.target_2d_lmks_y], axis=1
            )

            self.tf_trans = tf.Variable(
                np.zeros((1, 3)), name="trans", dtype=tf.float64, trainable=True
            )
            self.tf_rot = tf.Variable(
                np.zeros((1, 3)), name="rot", dtype=tf.float64, trainable=True
            )
            self.tf_pose = tf.Variable(
                np.zeros((1, 12)), name="pose", dtype=tf.float64, trainable=True
            )
            self.tf_shape = tf.Variable(
                np.zeros((1, 300)), name="shape", dtype=tf.float64, trainable=True
            )
            self.tf_exp = tf.Variable(
                np.zeros((1, 100)), name="expression", dtype=tf.float64, trainable=True
            )

            # tf_scale = tf.Variable(0, dtype=tf.float64)

            smpl = SMPL(template_path)

            self.tf_model = tf.squeeze(
                smpl(
                    self.tf_trans,
                    tf.concat((self.tf_shape, self.tf_exp), axis=-1),
                    tf.concat((self.tf_rot, self.tf_pose), axis=-1),
                )
            )

            lmks_3d = self.tf_get_model_lmks(self.tf_model)

            self.s2d = tf.reduce_mean(
                tf.linalg.norm(
                    self.target_2d_lmks - tf.reduce_mean(self.target_2d_lmks, axis=0),
                    axis=1,
                )
            )
            self.s3d = tf.reduce_mean(
                tf.sqrt(
                    tf.reduce_sum(
                        tf.square(lmks_3d - tf.reduce_mean(lmks_3d, axis=0))[:, :2],
                        axis=1,
                    )
                )
            )

            self.tf_scale = tf.Variable(self.s2d / self.s3d, dtype=lmks_3d.dtype)

            self.lmks_proj_2d = tf_project_points(lmks_3d, self.tf_scale, np.zeros(2))

            factor = tf.math.maximum(
                tf.math.reduce_max(self.target_2d_lmks[:, 0])
                - tf.math.reduce_min(self.target_2d_lmks[:, 0]),
                tf.math.reduce_max(self.target_2d_lmks[:, 1])
                - tf.math.reduce_min(self.target_2d_lmks[:, 1]),
            )

            self.lmk_dist = (
                weights["lmk"]
                * tf.reduce_sum(
                    tf.square(tf.subtract(self.lmks_proj_2d, self.target_2d_lmks))
                )
                / (factor ** 2)
            )

            self.neck_pose_reg = weights["neck_pose"] * tf.reduce_sum(
                tf.square(self.tf_pose[:3])
            )
            self.jaw_pose_reg = weights["jaw_pose"] * tf.reduce_sum(
                tf.square(self.tf_pose[3:6])
            )
            self.eyeballs_pose_reg = weights["eyeballs_pose"] * tf.reduce_sum(
                tf.square(self.tf_pose[6:])
            )
            self.shape_reg = weights["shape"] * tf.reduce_sum(tf.square(self.tf_shape))
            self.exp_reg = weights["expr"] * tf.reduce_sum(tf.square(self.tf_exp))

            self.optimizer1 = scipy_pt(
                loss=self.lmk_dist,
                var_list=[self.tf_scale, self.tf_trans, self.tf_rot],
                method="L-BFGS-B",
                options={"disp": 0, "ftol": 5e-6},
            )

            loss = (
                self.lmk_dist
                + self.shape_reg
                + self.exp_reg
                + self.neck_pose_reg
                + self.jaw_pose_reg
                + self.eyeballs_pose_reg
            )

            self.optimizer2 = scipy_pt(
                loss=loss,
                var_list=[
                    self.tf_scale,
                    self.tf_trans[:2],
                    self.tf_rot,
                    self.tf_pose,
                    self.tf_shape,
                    self.tf_exp,
                ],
                method="L-BFGS-B",
                options={"disp": 0, "ftol": 1e-7},
            )

    def tf_get_model_lmks(self, tf_model):
        """Get a differentiable landmark embedding in the FLAME surface"""

        lmk_face_idx, lmk_b_coords = load_embedding(
            BASE_DIR / CONFIG["flame"]["static_landmark_embedding_path"]
        )
        faces = tf.cast(
            tf.gather_nd(self.template_mesh, [[x] for x in lmk_face_idx.tolist()]),
            tf.int32,
        )
        return tf.einsum(
            "ijk,ij->ik", tf.gather(tf_model, faces), tf.convert_to_tensor(lmk_b_coords)
        )

    def fit_lmk2d_v2(self, pose, shape, expression, target_2d_lmks, file_name):

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.graph_options.optimizer_options.global_jit_level = (
            tf.compat.v1.OptimizerOptions.OFF
        )

        with tf.compat.v1.Session(config=config, graph=self.graph1) as session:
            session.run(tf.compat.v1.global_variables_initializer())

            self.tf_shape.load(
                np.expand_dims(np.hstack((shape, np.zeros(200))), 0), session,
            )

            self.tf_exp.load(
                np.expand_dims(np.hstack((expression, np.zeros(50))), 0), session,
            )

            self.tf_pose.load(
                np.expand_dims(np.hstack((pose[3:], np.zeros(9))), 0), session,
            )
            self.tf_rot.load(np.expand_dims(pose[:3], 0), session)

            self.target_2d_lmks_x.load(target_2d_lmks[:, :1], session)
            self.target_2d_lmks_y.load(target_2d_lmks[:, 1:], session)

            self.tf_scale.initializer.run()

            self.optimizer1.minimize(
                session,
                fetches=[
                    self.tf_model,
                    self.tf_scale,
                    self.template_mesh,
                    self.target_2d_lmks,
                    self.lmks_proj_2d,
                ],
            )

            self.optimizer2.minimize(
                session,
                fetches=[
                    self.tf_model,
                    self.tf_scale,
                    self.template_mesh,
                    self.target_2d_lmks,
                    self.lmks_proj_2d,
                    self.lmk_dist,
                    self.shape_reg,
                    self.exp_reg,
                    self.neck_pose_reg,
                    self.jaw_pose_reg,
                    self.eyeballs_pose_reg,
                ],
            )

            return (
                file_name,
                {
                    "tf_trans": self.tf_trans.eval(),
                    "tf_rot": self.tf_rot.eval(),
                    "tf_pose": self.tf_pose.eval(),
                    "tf_shape": self.tf_shape.eval(),
                    "tf_exp": self.tf_exp.eval(),
                },
            )


def extract_flame(fps):
    files = list(DATASET_DIR.glob(f"*/*/video_{fps}fps.mp4"))
    results = []
    for i, video_file in enumerate(
        tqdm(files, desc="Extracting flame parameters", leave=False)
    ):
        flame_dir = video_file.parent / f"flame_{fps}fps"
        gender = get_gender(video_file.parent.parent.name, video_file.parent.name)
        template_path = BASE_DIR / CONFIG["flame"][f"model_path_{gender}"]
        # with open(template_model_fname, "rb") as f:
        #     template = pickle.load(f, encoding="latin1")

        ringnet_file = video_file.parent / f"ringnet_{fps}fps.h5"
        openface_file = video_file.parent / f"openface_{fps}fps.csv"
        neutral_mesh_faces = Mesh(
            filename=str(video_file.parent / "neutral_mesh.ply")
        ).f

        f = h5py.File(ringnet_file, "r")["flame_params"]

        pool = ActorPool(
            [
                FrameOptimizer.remote(neutral_mesh_faces, template_path)
                for _ in range(32)
            ]
        )

        openface_data = list(openface_file.read_text().split("\n"))[1:]
        data = f["pose"], f["shape"], f["expression"], openface_data
        flame_dir.mkdir(parents=True, exist_ok=True)
        runners = []
        for i, (pose, shape, expression, openface) in enumerate(zip(*data), 1):
            flame_file = flame_dir / f"{i:06}.npy"
            if flame_file.exists():
                continue

            # Get 68 facial landmarks
            landmarks = [float(x) for x in openface.split(", ")[299:435]]
            # reshape the landmarks so that they are 2x51 (cut of the jaw (17 landmarks))
            target_2d_lmks = np.array(landmarks).reshape(2, -1).T[17:]
            runners.append((pose, shape, expression, target_2d_lmks, flame_file))

        for file_name, flame_params in tqdm(
            pool.map(lambda a, v: a.fit_lmk2d_v2.remote(*v), runners),
            total=len(runners),
            leave=False,
        ):
            np.save(file_name, flame_params)
