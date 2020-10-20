import gc
import io
import tempfile
from pathlib import Path

import numpy as np

import ffmpeg
from fastapi import Body, FastAPI, Request
from fastapi.responses import StreamingResponse
from visualize.faces import render_double_face_video, get_vert2

app = FastAPI()

VIDEO_DIR = "videos"


@app.post("/render")
def read_root(request: Request, data=Body(...)):
    seqs = io.BytesIO()
    seqs.write(data["seqs"].encode("latin-1"))
    seqs.seek(0)
    np_seqs = np.load(seqs)
    file_name = VIDEO_DIR / Path(data["file_name"])

    left_vert = get_vert2(
        np_seqs[0],
        exp_dim=data["exp_dim"],
        jaw_dim=data["jaw_dim"],
        neck_dim=data["neck_dim"],
    )
    right_vert = get_vert2(
        np_seqs[1],
        exp_dim=data["exp_dim"],
        jaw_dim=data["jaw_dim"],
        neck_dim=data["neck_dim"],
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmpf:
        render_double_face_video(tmpf.name, left_vert, right_vert, fps=25)
        file_name.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg.input(tmpf.name).output(str(file_name), vcodec="h264").run(quiet=True)
    gc.collect()
    url = f"http://{request.url.netloc}/video/{file_name}"
    return {"url": url}


@app.get("/video/{path:path}")
def read_item(request: Request, path: str):
    if not path.startswith(VIDEO_DIR):
        path = VIDEO_DIR / Path(path)
    return StreamingResponse(open(path, mode="rb"), media_type="video/mp4")
