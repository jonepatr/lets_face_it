import json
from misc.shared import BASE_DIR
from pathlib import Path
import re
from datetime import datetime


def get_gender(session, participant):
    meta_data = json.load((BASE_DIR / "data" / "meta_data.json").open())
    subject_id = meta_data["sessions"][session][participant]
    return meta_data["subjects"][subject_id]["gender"]


def get_participant(path):
    return re.search("\d_(.+)_FaceNear", path).group(1)


def replace_part(path: Path, original: str, replacement: str):
    return Path(*[x.replace(original, replacement) for x in path.parts])


def ms2frames(ms, fps=50):
    return round((ms / 1000) * fps) + 1


def frames2s(f, fps=50):
    return f / fps


def frames2ms(f, fps=50):
    return int(((f - 1) / fps) * 1000)


def get_training_name():
    dt = datetime.now()
    return f"{dt.day}-{dt.month}_{dt.hour}-{dt.minute}-{dt.second}.{str(dt.microsecond)[:2]}"

def get_face_indicies(exp_dim, jaw_dim, neck_dim, offset=0):
    p1_expression = list(range(offset, offset + exp_dim))
    p1_jaw = list(range(100 + offset, 100 + offset + jaw_dim))
    p1_neck = list(range(103 + offset, 103 + offset + neck_dim))

    return p1_expression + p1_jaw + p1_neck
