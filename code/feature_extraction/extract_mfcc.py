import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig
from python_speech_features import mfcc
from tqdm import tqdm

import ffmpeg
from misc.shared import DATA_DIR
from misc.utils import replace_part

MFCC_INPUTS = 26  # How many features we will store for each MFCC vector
WINDOW_LENGTH = 0.02
WINDOW_STEP = 0.01
NFFT = 1024

for file_ in tqdm(list(DATA_DIR.glob("Sessions_wav/*/*.wav"))):
    fs, audio = wav.read(file_)

    # Calculate MFCC feature with the window frame it was designed for
    mfcc_features = mfcc(
        audio,
        winlen=WINDOW_LENGTH,
        winstep=WINDOW_STEP,
        samplerate=fs,
        numcep=MFCC_INPUTS,
        nfft=NFFT,
    )

    # Get number of video frames
    video_name = next(
        replace_part(file_, "Sessions_wav", "Sessions_50fps").parent.glob(
            f"*{file_.stem}*.mp4"
        )
    )
    nb_frames = int(ffmpeg.probe(video_name)["streams"][0]["nb_frames"])

    # Resample to desired frame length
    resampled_sig = sig.resample(mfcc_features, nb_frames)

    dest_dir = replace_part(file_, "Sessions_wav", "Sessions_50fps_mfcc").with_suffix(
        ".npy"
    )
    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    np.save(dest_dir, resampled_sig)
