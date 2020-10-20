from psbody.mesh import Mesh
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import scipy.signal
from scipy.io import wavfile
from voca.utils.audio_handler import AudioHandler
from misc.shared import CONFIG, BASE_DIR


def inference(tf_model_fname, ds_fname, audio_fname, template, condition_idx):
    sample_rate, audio = wavfile.read(audio_fname)
    assert audio.ndim == 1, "Audio has multiple channels"

    audio_handler = AudioHandler(
        {
            "deepspeech_graph_fname": ds_fname,
            "audio_feature_type": "deepspeech",
            "num_audio_features": 29,
            "audio_window_size": 16,
            "audio_window_stride": 1,
        }
    )

    processed_audio = audio_handler.process(
        {"subj": {"seq": {"audio": audio, "sample_rate": sample_rate}}}
    )["subj"]["seq"]["audio"]

    # Load previously saved meta graph in the default graph
    saver = tf.train.import_meta_graph(tf_model_fname + ".meta")
    graph = tf.get_default_graph()

    speech_features = graph.get_tensor_by_name("VOCA/Inputs_encoder/speech_features:0")
    condition_subject_id = graph.get_tensor_by_name(
        "VOCA/Inputs_encoder/condition_subject_id:0"
    )
    is_training = graph.get_tensor_by_name("VOCA/Inputs_encoder/is_training:0")
    input_template = graph.get_tensor_by_name(
        "VOCA/Inputs_decoder/template_placeholder:0"
    )
    output_decoder = graph.get_tensor_by_name("VOCA/output_decoder:0")

    num_frames = processed_audio.shape[0]

    feed_dict = {
        speech_features: np.expand_dims(np.stack(processed_audio), -1),
        condition_subject_id: np.repeat(condition_idx - 1, num_frames),
        is_training: False,
        input_template: np.repeat(
            template.v[np.newaxis, :, :, np.newaxis], num_frames, axis=0
        ),
    }

    with tf.Session() as session:
        # Restore trained model
        saver.restore(session, tf_model_fname)
        predicted_vertices = np.squeeze(session.run(output_decoder, feed_dict))
    return predicted_vertices


def voca(
    audio_files,
    template_fname,
    dest_path,
    frame_count,
    condition_idx=3,
    tf_model_fname=BASE_DIR / CONFIG["voca"]["tf_model"],
    ds_fname=BASE_DIR / CONFIG["voca"]["ds_model"],
):
    if dest_path.exists():
        return True

    template = Mesh(filename=str(template_fname))
    all_meshes = []
    for audio_fname in tqdm(audio_files):
        results = inference(
            str(tf_model_fname),
            str(ds_fname),
            str(audio_fname),
            template,
            condition_idx,
        )
        if results.ndim == 2:
            results = np.expand_dims(results, 0)
        all_meshes.append(results)
        tf.reset_default_graph()
        tf.keras.backend.clear_session()

    meshes = np.vstack(all_meshes)

    fifty_fps_meshes = scipy.signal.resample(meshes, frame_count)
    dest_path.mkdir(parents=True, exist_ok=True)

    for i, predicted_verices in enumerate(tqdm(fifty_fps_meshes, desc="voca"), 1):
        np.save(dest_path / f"{i:05}.npy", predicted_verices)
