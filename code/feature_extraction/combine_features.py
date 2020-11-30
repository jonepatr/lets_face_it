import csv
import json
from typing import List

import h5py
import numpy as np
import ray
from misc.shared import BASE_DIR, DATASET_DIR, DATA_DIR
from misc.utils import ms2frames
from scipy.signal import savgol_filter
from collections import defaultdict

from tqdm import tqdm

WIN_LEN = 9


def prepare_openface(frame):
    return [float(col.strip()) for col in frame[299:435]]


def prepare_success(frame):
    return float(frame[3]) >= 0.98 and bool(frame[4])


def prepare_flame(flame):
    zeros = np.zeros((len(flame["tf_pose"]), 3))
    return {
        "expression": flame["tf_exp"][()],
        "jaw": flame["tf_pose"][:, 3:6],
        "neck": flame["tf_pose"][:, :3] + flame["tf_rot"],
        "rotation": flame["tf_rot"][()],
    }


def load_features(session_path, fps):
    participant_data = {"P1": {}, "P2": {}}

    for participant in ["P1", "P2"]:
        participant_path = session_path / participant

        p = participant_data[participant]

        openface_file = participant_path / f"openface_{fps}fps.csv"
        openface = list(csv.reader(openface_file.open()))[1:]
        p["openface"] = np.array([prepare_openface(frame) for frame in openface])
        p["success"] = [prepare_success(frame) for frame in openface]

        flame = h5py.File(participant_path / f"flame_{fps}fps.h5", "r")
        p["flame"] = prepare_flame(flame)
        p["flame"]["neck"] -= p["flame"]["rotation"][p["success"]].mean()

        p["mfcc"] = np.load(participant_path / f"mfcc_{fps}fps.npy")
        p["prosody"] = np.load(participant_path / f"prosodic_features_{fps}fps.npy")

        assert (
            len(p["flame"]["expression"])
            == len(p["success"])
            == len(p["openface"])
            == len(p["mfcc"])
            == len(p["prosody"])
        )
    return participant_data


def try_get(x, n):
    try:
        return n if x[n] else None
    except IndexError:
        return None


def get_with_preference(x, n1, n2, score=1):
    prev_1 = try_get(x, n1)
    if prev_1 is not None:
        return prev_1, 1
    elif score == 1:
        return try_get(x, n2), 2
    else:
        return None, -1


def get_frames(frame, success):
    if success[frame]:
        return frame
    else:
        prev, prev_score = get_with_preference(success, frame - 1, frame - 2)
        future, future_score = get_with_preference(
            success, frame + 1, frame + 2, prev_score
        )

        if prev and future:
            return [prev, future, 1 + prev_score + future_score, prev_score]
        else:
            return None


def get_frame_or_interpolate(frame, data):
    if isinstance(frame, int):
        result = data[frame]
    else:
        past, future, steps, current = frame
        result = np.linspace(data[past], data[future], steps, axis=0)[current]
    return result


def create_bins(participant_data, start, stop, agent, interlocutor):
    session_bins = []
    create_new_bin = True

    for frame in range(start, stop):
        agent_frame = get_frames(frame, participant_data[agent]["success"])
        p2_frame = get_frames(frame, participant_data[interlocutor]["success"])

        if agent_frame is not None and p2_frame is not None:
            if create_new_bin:
                session_bins.append([])
                create_new_bin = False
            session_bins[-1].append(
                [frame, (agent, agent_frame), (interlocutor, p2_frame)]
            )
        else:
            create_new_bin = True
    return session_bins


def save_segment(participant_data, start, stop, agent, interlocutor, win_len):
    session_bins = create_bins(participant_data, start, stop, agent, interlocutor)

    big_participant_data = {
        "agent": defaultdict(list),
        "interlocutor": defaultdict(list),
    }

    for session_bin in session_bins:
        new_participant_data = {
            agent: defaultdict(list),
            interlocutor: defaultdict(list),
        }

        # win_len is our smoothing window
        if len(session_bin) < win_len:
            continue
        for orig_frame, agent_frames, interlocutor_frames in session_bin:
            for p, frame in (agent_frames, interlocutor_frames):
                for data_name in ("mfcc", "prosody"):
                    result = participant_data[p][data_name][orig_frame]
                    new_participant_data[p][data_name].append(result)

                openface_data = participant_data[p]["openface"]
                result = get_frame_or_interpolate(frame, openface_data)
                new_participant_data[p]["openface"].append(result)

                flame_data = participant_data[p]["flame"]
                for data_name in ("jaw", "expression", "neck", "rotation"):
                    result = get_frame_or_interpolate(frame, flame_data[data_name])
                    new_participant_data[p][f"flame_{data_name}"].append(result)

        for who, p in (("agent", agent), ("interlocutor", interlocutor)):
            for key, value in new_participant_data[p].items():
                if key in ("mfcc", "prosody"):
                    smooth_data = value
                else:
                    smooth_data = savgol_filter(np.array(value), win_len, 3, axis=0)

                big_participant_data[who][key].append(smooth_data)

    frames = [[y[0] for y in x] for x in session_bins if len(x) >= win_len]
    return big_participant_data, frames


def combine_features(fps, win_len=9):
    data = json.loads((BASE_DIR / "data/train_val_test.json").read_text())
    stds = {}
    means = {}

    with h5py.File(DATA_DIR / "lets_face_it.h5", "w") as f:
        for data_type in ("train", "val", "test"):
            grand_output = defaultdict(lambda: defaultdict(list))

            for session, segments in tqdm(data.get(data_type, {}).items()):
                participant_data = load_features(DATASET_DIR / session, fps)
                for start, stop in segments:
                    for agent, interlocutor in [["P1", "P2"], ["P2", "P1"]]:
                        output, frames = save_segment(
                            participant_data,
                            ms2frames(start, fps) - 1,
                            ms2frames(stop, fps) - 1,
                            agent,
                            interlocutor,
                            win_len=win_len,
                        )
                        for p, output_data in output.items():
                            for data_kind, inner_data in output_data.items():
                                grand_output[data_kind][p] += inner_data

            # prep stds/means for standardization
            if data_type == "train":
                for key, value in grand_output.items():
                    rows = np.vstack([item for x in value["agent"] for item in x])
                    stds[key] = rows.std(axis=0)
                    means[key] = rows.mean(axis=0)
                    f.create_dataset(f"/stds/{key}", data=stds[key])
                    f.create_dataset(f"/means/{key}", data=means[key])

            for data_kind, value in grand_output.items():
                for p, sub_value in value.items():
                    for i, group in enumerate(sub_value):
                        if data_kind not in ("mfcc", "prosody"):
                            result = (group - means[data_kind]) / stds[data_kind]
                        else:
                            result = group

                        f.create_dataset(
                            f"/{data_type}/{data_kind}/{i}/{p}", data=result
                        )


if __name__ == "__main__":
    combine_features(25)