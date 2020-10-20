import itertools
import logging
import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from glob import glob
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
import numpy as np
import ray
from tensorflow.contrib.opt import ScipyOptimizerInterface as scipy_pt
from ray.util import ActorPool


ray.init(address="IP.IP.IP.IP", redis_password="XXXXXXXXXXXXXX")


def batch_skew(vec, batch_size=None, name=""):
    """
    vec is N x 3, batch_size is int

    returns N x 3 x 3. Skew_sym version of each matrix.
    """
    with tf.name_scope(name, "batch_skew", [vec]):
        if batch_size is None:
            batch_size = vec.shape.as_list()[0]
        col_inds = tf.constant([1, 2, 3, 5, 6, 7])
        indices = tf.reshape(
            tf.reshape(tf.range(0, batch_size) * 9, [-1, 1]) + col_inds, [-1, 1]
        )
        updates = tf.reshape(
            tf.stack(
                [-vec[:, 2], vec[:, 1], vec[:, 2], -vec[:, 0], -vec[:, 1], vec[:, 0]],
                axis=1,
            ),
            [-1],
        )
        out_shape = [batch_size * 9]
        res = tf.scatter_nd(indices, updates, out_shape)
        res = tf.reshape(res, [batch_size, 3, 3])

        return res


def batch_global_rigid_transformation(Rs, Js, parent, rotate_base=False, name=""):
    """
    Computes absolute joint locations given pose.

    rotate_base: if True, rotates the global rotation by 90 deg in x axis.
    if False, this is the original SMPL coordinate.

    Args:
      Rs: N x num_joints x 3 x 3 rotation vector of K joints
      Js: N x num_joints x 3, joint locations before posing
      parent: num_joints holding the parent id for each index

    Returns
      new_J : `Tensor`: N x num_joints x 3 location of absolute joints
      A     : `Tensor`: N x num_joints 4 x 4 relative joint transformations for LBS.
    """

    num_joints = parent.shape[0]
    with tf.name_scope(name, "batch_forward_kinematics", [Rs, Js]):
        N = Rs.shape[0].value
        if rotate_base:
            print("Flipping the SMPL coordinate frame!!!!")
            rot_x = tf.constant([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=Rs.dtype)
            rot_x = tf.reshape(tf.tile(rot_x, [N, 1]), [N, 3, 3])
            root_rotation = tf.matmul(Rs[:, 0, :, :], rot_x)
        else:
            root_rotation = Rs[:, 0, :, :]

        # Now Js is N x num_joints x 3 x 1
        Js = tf.expand_dims(Js, -1)

        def make_A(R, t, name=""):
            # Rs is N x 3 x 3, ts is N x 3 x 1
            with tf.name_scope(name, "Make_A", [R, t]):
                R_homo = tf.pad(R, [[0, 0], [0, 1], [0, 0]])
                t_homo = tf.concat([t, tf.ones([N, 1, 1], dtype=t.dtype)], 1)
                return tf.concat([R_homo, t_homo], 2)

        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]
        for i in range(1, parent.shape[0]):
            j_here = Js[:, i] - Js[:, parent[i]]
            A_here = make_A(Rs[:, i], j_here)
            res_here = tf.matmul(results[parent[i]], A_here, name="propA%d" % i)
            results.append(res_here)

        results = tf.stack(results, axis=1)

        new_J = results[:, :, :3, 3]

        # --- Compute relative A: Skinning is based on
        # how much the bone moved (not the final location of the bone)
        # but (final_bone - init_bone)
        # ---
        Js_w0 = tf.concat([Js, tf.zeros([N, num_joints, 1, 1], dtype=Js.dtype)], 2)
        init_bone = tf.matmul(results, Js_w0)
        # Append empty 4 x 3:
        init_bone = tf.pad(init_bone, [[0, 0], [0, 0], [0, 0], [3, 0]])
        A = results - init_bone

        return new_J, A


def batch_rodrigues(theta, name="", dtype=tf.float64):
    """
    Theta is N x 3
    """
    with tf.name_scope(name, "batch_rodrigues", [theta]):
        batch_size = theta.shape.as_list()[0]
        angle = tf.expand_dims(
            tf.clip_by_value(tf.norm(theta, axis=1), 1e-16, 1e16), -1
        )
        r = tf.expand_dims(tf.math.divide_no_nan(theta, angle), -1)

        angle = tf.expand_dims(angle, -1)
        cos = tf.cos(angle)
        sin = tf.sin(angle)

        outer = tf.matmul(r, r, transpose_b=True, name="outer")
        eyes = tf.tile(tf.expand_dims(tf.eye(3, dtype=dtype), 0), [batch_size, 1, 1])
        R = cos * eyes + (1 - cos) * outer + sin * batch_skew(r, batch_size=batch_size)
        return R


# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r


class SMPL(object):
    def __init__(self, dd, joint_type="cocoplus", dtype=tf.float64):
        """
        pkl_path is the path to a SMPL model
        """
        # -- Load SMPL params --

        self.dtype = dtype
        # Mean template vertices
        self.v_template = tf.Variable(
            undo_chumpy(dd["v_template"]),
            name="v_template",
            dtype=self.dtype,
            trainable=False,
        )

        # Size of mesh [Number of vertices, 3]
        self.size = [self.v_template.shape[0].value, 3]
        self.num_betas = dd["shapedirs"].shape[-1]
        self.num_verts = dd["shapedirs"]
        self.num_joints = dd["J"].shape[0]

        # Shape blend shape basis: num_verts x 3 x num_betas
        # reshaped to 3*num_verts x num_betas, transposed to num_betas x 3*num_verts
        shapedir = np.reshape(undo_chumpy(dd["shapedirs"]), [-1, self.num_betas]).T
        self.shapedirs = tf.Variable(
            shapedir, name="shapedirs", dtype=self.dtype, trainable=False
        )

        # Regressor for joint locations given shape - 6890 x 24
        self.J_regressor = tf.Variable(
            dd["J_regressor"].T.todense(),
            name="J_regressor",
            dtype=self.dtype,
            trainable=False,
        )

        # Pose blend shape basis: num_verts x 3 x 9*num_joints, reshaped to 3*num_verts x 9*num_joints
        num_pose_basis = dd["posedirs"].shape[-1]
        posedirs = np.reshape(undo_chumpy(dd["posedirs"]), [-1, num_pose_basis]).T
        self.posedirs = tf.Variable(
            posedirs, name="posedirs", dtype=self.dtype, trainable=False
        )

        # indices of parents for each joints
        self.parents = dd["kintree_table"][0].astype(np.int32)

        # LBS weights
        self.weights = tf.Variable(
            undo_chumpy(dd["weights"]),
            name="lbs_weights",
            dtype=self.dtype,
            trainable=False,
        )

    def __call__(self, trans, beta, theta, name=""):
        """
        Obtain SMPL with shape (beta) & pose (theta) inputs.
        Theta includes the global rotation.
        Args:
          beta: N x num_betas
          theta: N x 3*num_joints (with 3-D axis-angle rep)

        Updates:
        self.J_transformed: N x num_joints x 3 joint location after shaping
                 & posing with beta and theta

        Returns:
          - Verts: N x num_verts x 3
        """

        with tf.name_scope(name, "smpl_main", [beta, theta]):
            num_batch = beta.shape[0].value

            # 1. Add shape blend shapes
            # (N x num_betas) x (num_betas x 3*num_verts) = N x num_verts x 3
            v_shaped = (
                tf.reshape(
                    tf.matmul(beta, self.shapedirs, name="shape_bs"),
                    [-1, self.size[0], self.size[1]],
                )
                + self.v_template
            )

            # 2. Infer shape-dependent joint locations.
            Jx = tf.matmul(v_shaped[:, :, 0], self.J_regressor)
            Jy = tf.matmul(v_shaped[:, :, 1], self.J_regressor)
            Jz = tf.matmul(v_shaped[:, :, 2], self.J_regressor)
            J = tf.stack([Jx, Jy, Jz], axis=2)

            # 3. Add pose blend shapes
            # N x num_joints x 3 x 3
            Rs = tf.reshape(
                batch_rodrigues(tf.reshape(theta, [-1, 3])), [-1, self.num_joints, 3, 3]
            )
            with tf.name_scope("lrotmin"):
                # Ignore global rotation.
                pose_feature = tf.reshape(
                    Rs[:, 1:, :, :] - tf.eye(3, dtype=self.dtype),
                    [-1, 9 * (self.num_joints - 1)],
                )

            # (N x 9*(num_joints-1))) x (9*(num_joints-1), 3*num_verts) -> N x num_verts x 3
            v_posed = (
                tf.reshape(
                    tf.matmul(pose_feature, self.posedirs),
                    [-1, self.size[0], self.size[1]],
                )
                + v_shaped
            )

            # 4. Get the global joint location
            self.J_transformed, A = batch_global_rigid_transformation(
                Rs, J, self.parents
            )

            # 5. Do skinning:
            # W is N x num_verts x num_joints
            W = tf.reshape(
                tf.tile(self.weights, [num_batch, 1]), [num_batch, -1, self.num_joints]
            )

            # (N x num_verts x num_joints) x (N x num_joints x 16)
            T = tf.reshape(
                tf.matmul(W, tf.reshape(A, [num_batch, self.num_joints, 16])),
                [num_batch, -1, 4, 4],
            )
            v_posed_homo = tf.concat(
                [v_posed, tf.ones([num_batch, v_posed.shape[1], 1], dtype=self.dtype)],
                2,
            )
            v_homo = tf.matmul(T, tf.expand_dims(v_posed_homo, -1))

            return tf.add(v_homo[:, :, :3, 0], trans)


def load_binary_pickle(filepath):
    with open(filepath, "rb") as f:
        if sys.version_info >= (3, 0):
            data = pickle.load(f, encoding="latin1")
        else:
            data = pickle.load(f)
    return data


def load_embedding(file_path):
    """ funciton: load landmark embedding, in terms of face indices and barycentric coordinates for corresponding landmarks
    note: the included example is corresponding to CMU IntraFace 49-point landmark format.
    """
    lmk_indexes_dict = load_binary_pickle(file_path)
    lmk_face_idx = lmk_indexes_dict["lmk_face_idx"].astype(np.uint32)
    lmk_b_coords = lmk_indexes_dict["lmk_b_coords"]
    return lmk_face_idx, lmk_b_coords


@ray.remote(num_gpus=0.01)
class FrameOptimizer(object):
    def __init__(
        self, landmark_dir, neutral_mesh_faces, dd, lmk_face_idx, lmk_b_coords,
    ):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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

            smpl = SMPL(dd)

            self.tf_model = tf.squeeze(
                smpl(
                    self.tf_trans,
                    tf.concat((self.tf_shape, self.tf_exp), axis=-1),
                    tf.concat((self.tf_rot, self.tf_pose), axis=-1),
                )
            )

            lmks_3d = self.tf_get_model_lmks(
                self.tf_model, self.template_mesh, lmk_face_idx, lmk_b_coords
            )

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

            self.lmks_proj_2d = self.tf_project_points(
                lmks_3d, self.tf_scale, np.zeros(2)
            )

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

    def tf_project_points(self, points, scale, trans):
        """
        weak perspective camera
        """
        return tf.scalar_mul(
            scale,
            tf.transpose(
                tf.linalg.matmul(
                    tf.eye(num_rows=2, num_columns=3, dtype=points.dtype),
                    points,
                    transpose_b=True,
                )
            )
            + trans,
        )

    def tf_get_model_lmks(self, tf_model, template_mesh, lmk_face_idx, lmk_b_coords):
        """Get a differentiable landmark embedding in the FLAME surface"""

        faces = tf.cast(
            tf.gather_nd(self.template_mesh, [[x] for x in lmk_face_idx.tolist()]),
            tf.int32,
        )
        return tf.einsum(
            "ijk,ij->ik", tf.gather(tf_model, faces), tf.convert_to_tensor(lmk_b_coords)
        )

    def fit_lmk2d_v2(self, flame_out_path, flame_parameters, target_2d_lmks):

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.graph_options.optimizer_options.global_jit_level = (
            tf.compat.v1.OptimizerOptions.OFF
        )

        with tf.compat.v1.Session(config=config, graph=self.graph1) as session:
            session.run(tf.compat.v1.global_variables_initializer())

            self.tf_shape.load(
                np.expand_dims(
                    np.hstack((flame_parameters["shape"], np.zeros(200))), 0
                ),
                session,
            )

            self.tf_exp.load(
                np.expand_dims(
                    np.hstack((flame_parameters["expression"], np.zeros(50))), 0
                ),
                session,
            )

            self.tf_pose.load(
                np.expand_dims(
                    np.hstack((flame_parameters["pose"][3:], np.zeros(9))), 0
                ),
                session,
            )
            self.tf_rot.load(np.expand_dims(flame_parameters["pose"][:3], 0), session)

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

            return [
                flame_out_path,
                {
                    "tf_trans": self.tf_trans.eval(),
                    "tf_rot": self.tf_rot.eval(),
                    "tf_pose": self.tf_pose.eval(),
                    "tf_shape": self.tf_shape.eval(),
                    "tf_exp": self.tf_exp.eval(),
                },
            ]

    def null(self):
        return None

    def kill(self):
        ray.actor.exit_actor()


def ranges(i):
    for a, b in itertools.groupby(enumerate(i), lambda pair: pair[1] - pair[0]):
        b = list(b)
        yield b[0][1], b[-1][1]


def get_existing_files_old(flame_fitting_dir):
    @ray.remote
    def check_remote_dirs():
        return [
            int(file_.parent.name)
            for file_ in flame_fitting_dir.glob("*/flame_params.npy")
        ]

    checks = []
    for x in ray.nodes():
        if not x.get("Alive"):
            continue
        for key, item in x["Resources"].items():
            if key.startswith("node:"):
                checks.append(check_remote_dirs.options(resources={key: item}).remote())

    data = set()
    for x in ray.get(checks):
        data |= set(str(y).zfill(5) for y in x)

    return data


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


@ray.remote
def null():
    return None


def main():

    from misc.shared import BASE_DIR, DATA_DIR
    from misc.utils import get_gender, replace_part
    from psbody.mesh import Mesh
    from tqdm import tqdm

    done_sessions = BASE_DIR / "code/feature_extraction/done_sessions.txt"
    ds = done_sessions.read_text().strip().split("\n")

    FLAME_LMK_PATH = BASE_DIR / "code/TF_FLAME/data/flame_static_embedding.pkl"

    FITTING_DIR = "Sessions_50fps_flame_fitting"

    lmk_face_idx, lmk_b_coords = load_embedding(FLAME_LMK_PATH)

    folders = list(sorted(DATA_DIR.glob("Sessions_50fps_openface_51_landmarks/*/*")))
    pbar = tqdm(folders)

    for dir_ in pbar:
        if str(dir_) in ds:
            continue

        flame_fitting_dir = replace_part(
            dir_, "Sessions_50fps_openface_51_landmarks", FITTING_DIR,
        )

        participant = re.search(r".*_(.+?)_FaceNear", dir_.name).group(1)
        gender = get_gender(dir_.parent.name, participant)
        template_model_fname = str(
            BASE_DIR / "models/flame_model/ch_models/" / f"{gender}_model.pkl"
        )

        ringnet_dir = replace_part(
            dir_, "Sessions_50fps_openface_51_landmarks", "Sessions_50fps_ringnet",
        )

        with open(template_model_fname, "rb") as f:
            dd = pickle.load(f, encoding="latin1")

        neutral_mesh_path = ringnet_dir / "neutral_mesh.ply"
        neutral_mesh_faces = Mesh(filename=str(neutral_mesh_path)).f

        flame_fitting_dir.mkdir(parents=True, exist_ok=True)

        run_files(
            pbar,
            flame_fitting_dir,
            ringnet_dir,
            dir_,
            neutral_mesh_faces,
            dd,
            lmk_face_idx,
            lmk_b_coords,
        )
        with done_sessions.open("a") as f:
            f.write(str(dir_) + "\n")


def run(a, landmark_path, flame_fitting_dir, ringnet_dir):
    ringnet_path = ringnet_dir / "flame_params" / os.path.basename(landmark_path)
    try:
        flame_parameters = np.load(
            ringnet_path, allow_pickle=True, encoding="latin1"
        ).item()

        flame_parameters["vertices"]
        flame_parameters["cam"]

        target_2d_lmks = np.load(landmark_path)
    except:
        return null.remote()
    flame_out_path = (
        flame_fitting_dir / os.path.basename(landmark_path)[:-4] / "flame_params.npy"
    )

    return a.fit_lmk2d_v2.remote(str(flame_out_path), flame_parameters, target_2d_lmks)


def run_files(
    pbar,
    flame_fitting_dir,
    ringnet_dir,
    dir_,
    neutral_mesh_faces,
    dd,
    lmk_face_idx,
    lmk_b_coords,
    attempt=0,
):
    from tqdm import tqdm

    existing_files = set(
        os.path.basename(os.path.dirname(x))
        for x in glob(str(flame_fitting_dir / "*/flame_params.npy"))
    )

    files = sorted(
        [
            x
            for x in glob(str(dir_ / "*"))
            if os.path.basename(x)[:-4] not in existing_files
        ]
    )
    counter = 0
    actors = []

    cpu_count = int(ray.available_resources()["CPU"]) - 2

    pbar.set_description(f"{dir_.parent.name}/{dir_.name} ({cpu_count} cpus)")
    for x in range(min(len(files), cpu_count)):
        actors.append(
            FrameOptimizer.remote(
                dir_, neutral_mesh_faces, dd, lmk_face_idx, lmk_b_coords,
            )
        )
    file_len = len(files)
    pool = ActorPool(actors)
    try:
        pbar2 = tqdm(
            pool.map_unordered(
                lambda a, v: run(a, v, flame_fitting_dir, ringnet_dir), files
            ),
            total=file_len,
        )

        for x in pbar2:
            pbar2.set_description(f"{dir_.parent.name}/{dir_.name} ({cpu_count} cpus)")
            counter += 1
            if x is not None:
                flame_out_path, flame_out_params = x
                os.makedirs(os.path.dirname(flame_out_path), exist_ok=True)
                np.save(flame_out_path, flame_out_params)

    except ray.exceptions.RayActorError:
        if attempt > 10:
            raise Exception("too many attempts")
        for actor in actors:
            ray.kill(actor)
        if counter > 0:
            attempt = 0
        else:
            attempt += 1
        run_files(
            pbar,
            flame_fitting_dir,
            ringnet_dir,
            dir_,
            neutral_mesh_faces,
            dd,
            lmk_face_idx,
            lmk_b_coords,
            attempt=attempt,
        )


if __name__ == "__main__":
    main()
