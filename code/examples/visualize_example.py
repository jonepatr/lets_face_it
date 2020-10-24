import io
import numpy as np
import json
import requests
import h5py


seq_len = 100
file_name = "output_video.mp4"
data_file = "flame_params.hdf5"


def byteify(x):
    memfile = io.BytesIO()
    np.save(memfile, x)
    memfile.seek(0)
    return memfile.read().decode("latin-1")


def get_face(x, seq_len):
    return {
        "expression": byteify(x["tf_exp"][:seq_len]),
        "pose": byteify(x["tf_pose"][:seq_len]),
        "shape": byteify(x["tf_shape"][:seq_len]),
        "rotation": byteify(x["tf_rot"][:seq_len]),
    }


with h5py.File(data_file, "r") as f:
    p1 = f["sessions/1/participants/P1"]
    p2 = f["sessions/1/participants/P2"]

    serialized = json.dumps(
        {
            "seqs": [get_face(p1, seq_len), get_face(p2, seq_len)],
            "file_name": file_name,
            "fps": 25,
        }
    )
try:
    resp = requests.post("http://localhost:8000/render", data=serialized, timeout=600)
    resp.raise_for_status()
    print(resp.json())
except requests.exceptions.HTTPError:
    print("render request: failed on the server..")
except requests.exceptions.Timeout:
    print("render request: timed out")
except requests.exceptions.ConnectionError:
    print("render request: connection error")
