import numpy as np
from misc.shared import DATA_DIR
from misc.utils import replace_part
from tqdm import tqdm
import ffmpeg
from data_processing.audio_utils import crosstalk_vad

for i, dir_ in enumerate(tqdm(list(DATA_DIR.glob("Sessions_wav/*")))):
    dest_path = replace_part(dir_, "Sessions_wav", "Sessions_vad")
    if (dest_path / "P1.npy").exists():
        continue

    video_path = replace_part(dir_, "Sessions_wav", "Sessions_50fps")
    p1_frames = int(
        ffmpeg.probe(next(video_path.glob("*P1*")))["streams"][0]["nb_frames"]
    )
    p2_frames = int(
        ffmpeg.probe(next(video_path.glob("*P2*")))["streams"][0]["nb_frames"]
    )

    assert p1_frames == p2_frames

    s1x, s2x = crosstalk_vad(
        dir_ / "P1.wav",
        dir_ / "P2.wav",
        p1_frames,
        tha=30,
        thb=5,
        savgol_win=301,
        savgol_poly_order=1,
    )

    dest_path.mkdir(parents=True, exist_ok=True)
    np.save(dest_path / "P1.npy", s1x)
    np.save(dest_path / "P2.npy", s2x)
