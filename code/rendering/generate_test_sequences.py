
import json
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from misc.shared import DATA_DIR, RANDOM_SEED
from visualize.faces import render_double_face_video, render_face

random.seed(RANDOM_SEED)
SHAPE_DIM = 300


def get_vad_weights(participant, session, start_frames, stop_frames):
    vad_weights = np.load(
        (DATA_DIR / "Sessions_vad" / session / participant).with_suffix(".npy")
    )
    assert start_frames > 1
    return np.expand_dims(vad_weights[start_frames - 1 : stop_frames : 2], 1)


def get_vocas(participant, session, frame_nbs, vad_scaling_factor=1):
    int_frame_nbs = list(map(int, frame_nbs))
    vad_weights = (
        get_vad_weights(participant, session, min(int_frame_nbs), max(int_frame_nbs))
        * vad_scaling_factor
    )

    voca_sessions = DATA_DIR / "Sessions_50fps_voca" / session
    voca_files = sorted(voca_sessions.glob(f"*{participant}*/flame_params/*"))

    poses = []
    expression = []
    for x in voca_files:
        if x.stem in frame_nbs:
            d = np.load(x, allow_pickle=True).item()
            poses.append(d["tf_pose"])
            expression.append(d["tf_exp"])

    return {
        "pose": torch.from_numpy(np.vstack(poses) * vad_weights).float(),
        "expression": torch.from_numpy(np.vstack(expression) * vad_weights).float(),
    }


def generate_videos(sequences, ouput_dir, vad_scaling_factor=1, overwrite=False):
    ouput_dir.mkdir(exist_ok=True, parents=True)

    for file_name, session, left_face, right_face, info, frame_nbs in tqdm(
        sequences, leave=False
    ):
        output_file = ouput_dir / file_name

        if output_file.exists() and not overwrite:
            continue

        seq_len = left_face["expression"].shape[0]

        if info:
            left_gender = info["left_gender"]
            right_gender = info["right_gender"]
            left_shape = (
                torch.tensor(info["left_shape"]).unsqueeze(0).repeat(seq_len, 1)
            )
            right_shape = (
                torch.tensor(info["right_shape"]).unsqueeze(0).repeat(seq_len, 1)
            )
            left_skin_color = info["left_skin_color"]
            right_skin_color = info["right_skin_color"]
            start = [info["left_start"], info["right_start"]]

        else:
            left_gender = random.choice(["male", "female"])
            right_gender = random.choice(["male", "female"])

            left_shape = torch.randn(SHAPE_DIM).repeat(seq_len, 1)
            right_shape = torch.randn(SHAPE_DIM).repeat(seq_len, 1)

            skin_colors = ["white", "black"]

            left_skin_color = random.choice(skin_colors)
            right_skin_color = random.choice(skin_colors)

            start = random.sample([0, 136], 2)

        left_participant = "P1" if start[0] == 0 else "P2"
        right_participant = "P1" if start[0] == 136 else "P2"

        left_lipsync = get_vocas(
            left_participant, session, frame_nbs, vad_scaling_factor
        )
        right_lipsync = get_vocas(
            right_participant, session, frame_nbs, vad_scaling_factor
        )

        vertices = render_face(
            left_gender, left_face, left_lipsync, right_side=False, shape=left_shape,
        )

        vertices2 = render_face(
            right_gender, right_face, right_lipsync, right_side=True, shape=right_shape,
        )

        if not info:
            meta_dir = output_file.parent / "meta"
            meta_dir.mkdir(exist_ok=True, parents=True)
            (meta_dir / output_file.stem).with_suffix(".txt").write_text(
                json.dumps(
                    {
                        "file_name": file_name,
                        "left_start": start[0],
                        "right_start": start[1],
                        "left_gender": left_gender,
                        "right_gender": right_gender,
                        "left_shape": left_shape[0].tolist(),
                        "right_shape": right_shape[0].tolist(),
                        "left_skin_color": left_skin_color,
                        "right_skin_color": right_skin_color,
                    }
                )
            )

        with tempfile.TemporaryDirectory() as tmpd:
            f_name = Path(tmpd) / file_name
            render_double_face_video(
                str(f_name),
                vertices,
                vertices2,
                fps=25,
                skin_color_v1=left_skin_color,
                skin_color_v2=right_skin_color,
                width=2048
            )
            shutil.move(f_name, output_file)
