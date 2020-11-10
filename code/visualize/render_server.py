import gc
import io
import os
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

import ffmpeg
import numpy as np
import torch
from fastapi import Body, FastAPI, Request
from fastapi.responses import StreamingResponse

try:
    subprocess.check_call(
        ["python3", Path(__file__).parent / "test_egl.py"], stderr=subprocess.DEVNULL
    )
    os.environ["PYOPENGL_PLATFORM"] = "egl"
except subprocess.CalledProcessError:
    print("Could not use EGL (GPU support), falling back on using CPU")
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

from visualize.render_tools import get_vertices, render_double_face_video

app = FastAPI()

VIDEO_DIR = "videos"


def debyteify(x, key):
    seqs = io.BytesIO()
    seqs.write(x[key].encode("latin-1"))
    seqs.seek(0)
    return torch.from_numpy(np.load(seqs)).float()


def get_vert(seq):
    return get_vertices(
        debyteify(seq, "expression"),
        debyteify(seq, "pose"),
        debyteify(seq, "rotation"),
        shape=debyteify(seq, "shape"),
    )


@app.post("/render")
def read_root(request: Request, data=Body(...)):
    file_name = VIDEO_DIR / Path(data.get("file_name", str(uuid4())))
    fps = data["fps"]
    left_vert = get_vert(data["seqs"][0])
    right_vert = get_vert(data["seqs"][1])

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmpf:
        render_double_face_video(tmpf.name, left_vert, right_vert, fps=fps)
        file_name.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg.input(tmpf.name).output(str(file_name), vcodec="h264").run(
            overwrite_output=True
        )
    gc.collect()
    url = f"http://{request.url.netloc}/video/{file_name}"
    return {"url": url}


@app.get("/video/{path:path}")
def read_item(request: Request, path: str):
    if not path.startswith(VIDEO_DIR):
        path = VIDEO_DIR / Path(path)
    return StreamingResponse(open(path, mode="rb"), media_type="video/mp4")
