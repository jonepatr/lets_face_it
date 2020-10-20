import random

import torch

from glow_pytorch.glow.lets_face_it_glow import LetsFaceItGlow
from glow_pytorch.glow.utils import get_longest_history
from data_segments.find_test_segments import get_frames
from misc.shared import DATA_DIR, RANDOM_SEED
from misc.utils import get_face_indicies
from pytorch_lightning import seed_everything
from visualize.faces import get_vert, render_double_face_video

# seed_everything(RANDOM_SEED)


def get_data(data, model, use_zero_pose=True):

    standardize = lambda x, m, s: ((x - m) / s).unsqueeze(0)

    p1_face_data = data["p1_face"][: get_longest_history(model.hparams.Conditioning)]

    if use_zero_pose:
        p1_face = torch.zeros_like(p1_face_data).unsqueeze(0)
    else:
        p1_face = standardize(p1_face_data, model.face_means, model.face_stds)

    return {
        "p1_face": p1_face,
        "p2_face": standardize(data["p2_face"], model.face_means, model.face_stds),
        "p1_speech": standardize(
            data["p1_speech"], model.speech_means, model.speech_stds
        ),
        "p2_speech": standardize(
            data["p2_speech"], model.speech_means, model.speech_stds
        ),
    }


def expand_face_dim(seq, data_hparams):
    exp_dim = data_hparams["expression_dim"]
    jaw_dim = data_hparams["jaw_dim"]
    neck_dim = data_hparams["neck_dim"]

    output = torch.zeros((seq.size(0), seq.size(1), 106))
    output[:, :, :exp_dim] = seq[:, :, :exp_dim]
    output[:, :, 100 : 100 + jaw_dim] = seq[:, :, exp_dim : exp_dim + jaw_dim]

    output[:, :, 103 : 103 + neck_dim] = seq[
        :, :, exp_dim + jaw_dim : exp_dim + jaw_dim + neck_dim
    ]
    return output


def generate_motion(frames, model_path, dataset_root=str(DATA_DIR), eps=1):

    model = LetsFaceItGlow.load_from_checkpoint(model_path, dataset_root=dataset_root)
    model.hparams.Infer["eps"] = eps
    model.eval()

    seq_len = frames.size(0)

    data = dictify_frames(frames, model.hparams.Data)

    cond_data = get_data(data, model, use_zero_pose=True)

    predicted_seq = model.inference(seq_len, data=cond_data)

    destandardized_ouput = predicted_seq * model.face_stds + model.face_means

    return expand_face_dim(destandardized_ouput, model.hparams.Data)


def dictify_frames(frames, data_hparams):
    exp_dim = data_hparams["expression_dim"]
    jaw_dim = data_hparams["jaw_dim"]
    neck_dim = data_hparams["neck_dim"]
    speech_dim = data_hparams["speech_dim"]

    left_face_indicies = get_face_indicies(exp_dim, jaw_dim, neck_dim)
    right_face_indicies = get_face_indicies(exp_dim, jaw_dim, neck_dim, offset=136)

    return {
        "p1_face": frames[:, left_face_indicies],
        "p1_speech": frames[:, 106 : 106 + speech_dim],
        "p2_face": frames[:, right_face_indicies],
        "p2_speech": frames[:, 242 : 242 + speech_dim],
    }


if __name__ == "__main__":
    frames = get_frames("18", 1268, 1668, type_="test")

    predicted_faces_p1 = generate_motion(frames, "models/final_model.ckpt")
    face_verts_p1 = get_vert(
        predicted_faces_p1[0], gender=random.choice(["male", "female"])
    )

    ddd = dictify_frames(
        frames, {"expression_dim": 100, "jaw_dim": 3, "neck_dim": 3, "speech_dim": 30}
    )

    face_verts_p2 = get_vert(
        ddd["p2_face"][-predicted_faces_p1[0].size(0) :],
        gender=random.choice(["male", "female"]),
    )

    render_double_face_video(
        f"outputs/testing12348.mp4",
        face_verts_p2,
        face_verts_p1,
        skin_color_v1=random.choice(["white", "black"]),
        skin_color_v2=random.choice(["white", "black"]),
        fps=25,
    )
