import contextlib
import os

import numpy as np
import scipy.signal

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow.contrib.opt.python.training.external_optimizer import (
        ScipyOptimizerInterface,
    )
from misc.shared import BASE_DIR, CONFIG, DATASET_DIR
from psbody.mesh import Mesh
from scipy.io import wavfile

from TF_FLAME.tf_smpl.batch_smpl import SMPL
from tqdm import tqdm

from feature_extraction.shared import count_video_frames
from voca.utils.audio_handler import AudioHandler


class MeshFitter:
    def __init__(self, template_fname):
        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        weights = {}
        # Weight of the data term
        weights["data"] = 1000.0
        # Weight of the shape regularizer (the lower, the less shape is constrained)
        weights["shape"] = 1e-4
        # Weight of the expression regularizer (the lower, the less expression is constrained)
        weights["expr"] = 1e-4
        # Weight of the neck pose (i.e. neck rotationh around the neck) regularizer (the lower, the less neck pose is constrained)
        weights["neck_pose"] = 1e-4
        # Weight of the jaw pose (i.e. jaw rotation for opening the mouth) regularizer (the lower, the less jaw pose is constrained)
        weights["jaw_pose"] = 1e-4
        # Weight of the eyeball pose (i.e. eyeball rotations) regularizer (the lower, the less eyeballs pose is constrained)
        weights["eyeballs_pose"] = 1e-4

        self.tf_trans = tf.Variable(
            np.zeros((1, 3)), name="trans", dtype=tf.float64, trainable=True
        )
        self.tf_rot = tf.Variable(
            np.zeros((1, 3)), name="pose", dtype=tf.float64, trainable=True
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
        smpl = SMPL(template_fname)
        tf_model = tf.squeeze(
            smpl(
                self.tf_trans,
                tf.concat((self.tf_shape, self.tf_exp), axis=-1),
                tf.concat((self.tf_rot, self.tf_pose), axis=-1),
            )
        )

        # self.target_mesh = tf.placeholder(tf.float64, shape=tf_model.shape, name="target_mesh")
        self.target_mesh = tf.Variable(tf.zeros_like(tf_model))

        mesh_dist = tf.reduce_sum(tf.square(tf.subtract(tf_model, self.target_mesh)))
        neck_pose_reg = tf.reduce_sum(tf.square(self.tf_pose[:3]))
        jaw_pose_reg = tf.reduce_sum(tf.square(self.tf_pose[3:6]))
        eyeballs_pose_reg = tf.reduce_sum(tf.square(self.tf_pose[6:]))
        shape_reg = tf.reduce_sum(tf.square(self.tf_shape))
        exp_reg = tf.reduce_sum(tf.square(self.tf_exp))

        self.optimizer1 = ScipyOptimizerInterface(
            loss=mesh_dist,
            var_list=[self.tf_trans, self.tf_rot],
            method="BFGS",
            options={"disp": 0},
        )

        loss = (
            weights["data"] * mesh_dist
            + weights["shape"] * shape_reg
            + weights["expr"] * exp_reg
            + weights["neck_pose"] * neck_pose_reg
            + weights["jaw_pose"] * jaw_pose_reg
            + weights["eyeballs_pose"] * eyeballs_pose_reg
        )

        self.optimizer2 = ScipyOptimizerInterface(
            loss=loss,
            var_list=[self.tf_trans, self.tf_pose, self.tf_shape, self.tf_exp],
            method="BFGS",
            options={"disp": 0},
        )

    def fit(self, target_mesh, dest_path):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.graph_options.optimizer_options.global_jit_level = (
            tf.compat.v1.OptimizerOptions.OFF
        )
        with tf.compat.v1.Session(config=config) as session:
            session.run(tf.compat.v1.global_variables_initializer())
            vertices = target_mesh.astype(np.float64)
            self.optimizer1.minimize(session, feed_dict={self.target_mesh: vertices})
            self.optimizer2.minimize(session, feed_dict={self.target_mesh: vertices})

            return [
                dest_path,
                {
                    "tf_trans": self.tf_trans.eval(),
                    "tf_rot": self.tf_rot.eval(),
                    "tf_pose": self.tf_pose.eval(),
                    "tf_shape": self.tf_shape.eval(),
                    "tf_exp": self.tf_exp.eval(),
                },
            ]


class VocaModel:
    def __init__(self):
        self.audio_handler = AudioHandler(
            {
                "deepspeech_graph_fname": str(BASE_DIR / CONFIG["voca"]["ds_model"]),
                "audio_feature_type": "deepspeech",
                "num_audio_features": 29,
                "audio_window_size": 16,
                "audio_window_stride": 1,
            }
        )
        # Load previously saved meta graph in the default graph
        saver = tf.train.import_meta_graph(
            str(BASE_DIR / CONFIG["voca"]["tf_model"]) + ".meta"
        )
        graph = tf.get_default_graph()
        self.speech_features = graph.get_tensor_by_name(
            "VOCA/Inputs_encoder/speech_features:0"
        )
        self.condition_subject_id = graph.get_tensor_by_name(
            "VOCA/Inputs_encoder/condition_subject_id:0"
        )
        self.is_training = graph.get_tensor_by_name("VOCA/Inputs_encoder/is_training:0")
        self.input_template = graph.get_tensor_by_name(
            "VOCA/Inputs_decoder/template_placeholder:0"
        )
        self.output_decoder = graph.get_tensor_by_name("VOCA/output_decoder:0")
        self.session = tf.Session()
        saver.restore(self.session, str(BASE_DIR / CONFIG["voca"]["tf_model"]))

    def inference(self, audio_fname, template, condition_idx=3):
        sample_rate, audio = wavfile.read(audio_fname)
        assert audio.ndim == 1, "Audio has multiple channels"

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            audio = {"subj": {"seq": {"audio": audio, "sample_rate": sample_rate}}}
            processed_audio = self.audio_handler.process(audio)["subj"]["seq"]["audio"]

        num_frames = processed_audio.shape[0]
        feed_dict = {
            self.speech_features: np.expand_dims(np.stack(processed_audio), -1),
            self.condition_subject_id: np.repeat(condition_idx - 1, num_frames),
            self.is_training: False,
            self.input_template: np.repeat(
                template.v[np.newaxis, :, :, np.newaxis], num_frames, axis=0
            ),
        }
        results = np.squeeze(self.session.run(self.output_decoder, feed_dict))
        tf.reset_default_graph()
        if results.ndim == 2:
            results = np.expand_dims(results, 0)
        return results


def extract_voca(fps):
    msg = "Extracting voca"
    voca_model = VocaModel()
    for participant in tqdm(list(DATASET_DIR.glob("*/*")), desc=msg, leave=False):
        voca_file = participant / f"voca_mesh_{fps}fps.npy"
        neutral_mesh = participant / "neutral_mesh.ply"
        if voca_file.exists() or not neutral_mesh.exists():
            continue

        audio_chunks = sorted(participant.glob("audio_chunks/*.wav"))
        template = Mesh(filename=str(neutral_mesh))
        nb_frames = count_video_frames(participant / f"video_{fps}fps.mp4")

        all_meshes = []
        for audio_fname in tqdm(audio_chunks, leave=True):
            results = voca_model.inference(str(audio_fname), template)
            all_meshes.append(results)

        voca_meshes = np.vstack(all_meshes)

        resampled_meshes = scipy.signal.resample(voca_meshes, nb_frames)

        np.save(voca_file, resampled_meshes)
