import numpy as np
import scipy.signal as sig
from tqdm import tqdm

import ffmpeg
from data_processing.audio_utils import extract_prosodic_features
from misc.shared import DATA_DIR
from misc.utils import replace_part

for file_ in tqdm(list(DATA_DIR.glob("Sessions_wav/*/*.wav"))):
    prosodic_features = extract_prosodic_features(str(file_), time_step=0.02)

    video_name = next(
        replace_part(file_, "Sessions_wav", "Sessions_50fps").parent.glob(
            f"*{file_.stem}*.mp4"
        )
    )
    nb_frames = int(ffmpeg.probe(video_name)["streams"][0]["nb_frames"])

    prosodic_features_resmapled = sig.resample(prosodic_features, nb_frames)

    dest_dir = replace_part(
        file_, "Sessions_wav", "Sessions_50fps_prosodic"
    ).with_suffix(".npy")
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    np.save(dest_dir, prosodic_features_resmapled)
