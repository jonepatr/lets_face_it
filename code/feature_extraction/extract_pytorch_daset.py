from collections import defaultdict
from tarfile import TarFile

import h5py
import numpy as np
import torch
from scipy.signal import savgol_filter
from tqdm import tqdm

import ffmpeg
import ray
from data_segments.get_data_segments import (get_flame_params_for_file,
                                             get_segments_v2)
from misc.read_n_write import flame2glow
from misc.shared import DATA_DIR
from misc.utils import frames2ms, get_participant

WIN_LEN = 9
ONLY_ODD = True

ray.init(num_cpus=10)

segments = get_segments_v2()

sessions_dir = DATA_DIR / "Sessions_50fps_pytorch_data/sessions"


def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


def get_openface(file_):

    tar_file = TarFile(file_)
    openface_data = {}

    d = tar_file.extractfile(
        [x for x in tar_file.getmembers() if x.path.endswith(".csv")][0]
    ).readlines()

    failed = set()
    reference = []
    for i, line in enumerate(d):
        split_line = line.decode("utf-8").strip().split(",")
        if i == 0:
            reference = {x: split_line.index(x) for x in split_line}
            continue

        frame = int(split_line[reference["frame"]])
        if not ONLY_ODD or frame % 2 == 1:
            confidence = float(split_line[reference["confidence"]])
            success = bool(split_line[reference["success"]])
            if not success or confidence < 0.98:
                failed.add(frame)

    return failed


def ok(x):
    return x is not None


@ray.remote
def save_segment(session_path, segment_data):
    session_data = {
        "train": {"P1": [], "P2": []},
        "test": {"P1": [], "P2": []},
        "val": {"P1": [], "P2": []},
    }
    rot_pitch_avg = {}

    video_path = next((DATA_DIR / "Sessions_50fps" / session_path.name).glob("*P1*"))
    nb_frames = int(ffmpeg.probe(video_path)["streams"][0]["nb_frames"])

    failed = {}
    rot_pitch_sum = {}
    mfcc_data = {}
    prosodic_data = {}
    flame_params_paths = {}
    used_frames = {}
    history = {}
    for participant_path in session_path.glob("*"):
        participant = get_participant(str(participant_path))
        openface_file = (
            DATA_DIR
            / "Sessions_50fps_openface"
            / session_path.name
            / f"{participant_path.name}.tar"
        )
        failed[participant] = get_openface(openface_file)
        mfcc_data[participant] = np.load(
            DATA_DIR / "Sessions_50fps_mfcc" / session_path.name / f"{participant}.npy"
        )
        prosodic_data[participant] = np.load(
            DATA_DIR
            / "Sessions_50fps_prosodic"
            / session_path.name
            / f"{participant}.npy"
        )
        flame_params_paths[participant] = session_path / participant_path.name

        rot_pitch_sum[participant] = 0
        used_frames[participant] = 0
        history[participant] = []

    def process_row(participant):
        if frame in failed[participant]:
            return None

        flame_param_path = (
            flame_params_paths[participant] / f"{frame:05}" / "flame_params.npy"
        )
        try:
            shape, expression, pose, neck, eye, rot = get_flame_params_for_file(
                flame_param_path
            )
        except FileNotFoundError:
            return None
        face_array = flame2glow(expression[None, :], pose[None, :], neck[None, :])[0]
        mfcc = mfcc_data[participant][frame - 1]
        prosodic = prosodic_data[participant][frame - 1]
        rot_pitch_sum[participant] += rot[0]
        used_frames[participant] += 1
        return np.concatenate([face_array, mfcc, prosodic])

    for frame in range(1, nb_frames + 1, 2):
        frame_ms = frames2ms(frame)

        try:
            type_ = [x[1] for x in segment_data if x[2] <= frame_ms <= x[3]][0]
        except IndexError:
            print(segment_data, session_path, frame_ms)

        result = {participant: process_row(participant) for participant in ("P1", "P2")}

        session_data[type_]["P1"].append(
            [result["P1"], result["P2"], np.array([frame])]
        )
        session_data[type_]["P2"].append(
            [result["P2"], result["P1"], np.array([frame])]
        )

    rot_pitch_avg["P1"] = rot_pitch_sum["P1"] / used_frames["P1"]
    rot_pitch_avg["P2"] = rot_pitch_sum["P2"] / used_frames["P2"]

    dest_dir = sessions_dir / session_path.name
    dest_dir.mkdir(parents=True, exist_ok=True)

    for type_, participant_session_bins in session_data.items():
        output = []
        for participant, session_rows in participant_session_bins.items():
            session_bins = [[]]
            saved_rows = []
            create_new_list = True
            session_rows = (
                [[None, None, None], [None, None, None]]
                + session_rows
                + [[None, None, None], [None, None, None]]
            )

            def get_interp_val(i, n):
                prev_2 = session_rows[i - 2][n]
                prev_1 = session_rows[i - 1][n]
                future_1 = session_rows[i + 1][n]
                future_2 = session_rows[i + 2][n]

                vals = None
                if ok(prev_1):
                    if ok(future_1):
                        vals = (prev_1, future_1, 3, 1)
                    elif ok(future_2):
                        vals = (prev_1, future_2, 4, 1)
                elif ok(prev_2):
                    if ok(future_1):
                        vals = (prev_2, future_1, 4, 2)

                if vals:
                    past, future, steps, current = vals
                    return np.linspace(past, future, steps, axis=0)[current]
                else:
                    return None

            create_new_bin = True

            for i in range(2, len(session_rows) - 2):
                c1, c2, frame = session_rows[i]

                d1 = c1 if ok(c1) else get_interp_val(i, 0)
                d2 = c2 if ok(c2) else get_interp_val(i, 1)

                if ok(d1) and ok(d2):
                    if create_new_bin:
                        session_bins.append([])
                        create_new_bin = False
                    session_bins[-1].append(np.hstack([d1, d2, frame]))
                else:
                    create_new_bin = True

            for session_bin in session_bins:
                # WIN_LEN is our smoothing window
                if len(session_bin) >= WIN_LEN:
                    session_bin_np = np.array(session_bin)
                    session_bin_np[:, 103] -= rot_pitch_avg[participant]  # neck
                    session_bin_np[:, 239] -= rot_pitch_avg[participant]  # neck

                    # import pdb; pdb.set_trace()
                    session_bin_np[:, :106] = savgol_filter(
                        session_bin_np[:, :106], WIN_LEN, 3, axis=0
                    )
                    session_bin_np[:, 136:242] = savgol_filter(
                        session_bin_np[:, 136:242], WIN_LEN, 3, axis=0
                    )

                    output.append(torch.from_numpy(session_bin_np).float())

        torch.save(output, dest_dir / f"{type_}_25fps.pt")


segments_by_session = defaultdict(list)
for segment_data in segments:
    segments_by_session[segment_data[0]].append(segment_data)

results = []
for session_path in list(DATA_DIR.glob("Sessions_50fps_flame_fitting/*")):
    if session_path.stem == "35" or (sessions_dir / session_path.name).exists():
        continue
    results.append(
        save_segment.remote(session_path, segments_by_session[session_path.stem])
        # save_segment(session_path, segments_by_session[session_path.stem])
    )

for result in tqdm(to_iterator(results), total=len(results)):
    pass

sequences = []
for segment_file in sessions_dir.glob(f"*/train_25fps.pt"):
    for chunk in torch.load(segment_file):
        sequences.append(chunk)

data = torch.cat(sequences)
fixed_data = data[:, list(range(136))]
means = fixed_data.mean(dim=0)
stds = fixed_data.std(dim=0)

p1_face = list(range(106))
p1_speech = list(range(106, 136))
p2_face = list(range(136, 242))
p2_speech = list(range(242, 272))
frame_nb = [272]

for type_ in ("train", "val", "test"):
    save_path = DATA_DIR / f"Sessions_50fps_pytorch_data/{type_}.hdf5"
    if save_path.exists():
        continue

    sequences = []
    chunks = []
    for segment_file in sessions_dir.glob(f"*/{type_}_25fps.pt"):
        for chunk in torch.load(segment_file):
            chunk = torch.cat([chunk[1:], chunk[1:, :106] - chunk[:1, :106],], dim=1)
            chunk[:, :136] = (chunk[:, :136] - means[:136]) / stds[:136]
            chunk[:, 136:272] = (chunk[:, 136:272] - means[:136]) / stds[:136]
            sequences.append(chunk)
            chunks.append(chunk.shape[0])

    dataset = torch.cat(sequences)

    with h5py.File(save_path, "w") as f:
        f.create_dataset("standardization/face/means", data=means[:106])
        f.create_dataset("standardization/face/stds", data=stds[:106])
        f.create_dataset("standardization/speech/means", data=means[106:136])
        f.create_dataset("standardization/speech/stds", data=stds[106:136])
        f.create_dataset("chunks", data=chunks)
        f.create_dataset("p1_face", data=dataset[:, p1_face])
        f.create_dataset("p2_face", data=dataset[:, p2_face])
        f.create_dataset("p1_speech", data=dataset[:, p1_speech])
        f.create_dataset("p2_speech", data=dataset[:, p2_speech])
        f.create_dataset("frame_nb", data=dataset[:, frame_nb])
