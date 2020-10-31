import contextlib
import os

import numpy as np
import scipy.signal
import tensorflow as tf
from misc.shared import BASE_DIR, CONFIG, DATASET_DIR
from psbody.mesh import Mesh
from scipy.io import wavfile
from tqdm import tqdm

from feature_extraction.shared import count_video_frames
from voca.utils.audio_handler import AudioHandler


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
