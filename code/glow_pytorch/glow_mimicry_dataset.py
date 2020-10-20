from torch.utils.data import Dataset
import torch
import h5py
import random
from misc.utils import get_face_indicies

class GlowMimicryDataset(Dataset):
    def __init__(
        self, file_name, data_hparams=None, conditioning_hparams=None, seq_len=None,
    ):
        super().__init__()
        self.file_name = file_name

        self.face_indicies = get_face_indicies(
            data_hparams["expression_dim"],
            data_hparams["jaw_dim"],
            data_hparams["neck_dim"],
        )

        self.speech_dim = data_hparams["speech_dim"]
        self.p1_speech_history = conditioning_hparams["p1_speech"]["history"]
        self.p2_speech_history = conditioning_hparams["p2_speech"]["history"]
        self.p2_face_history = conditioning_hparams["p2_face"]["history"]

        self.use_frame_nb = conditioning_hparams["use_frame_nb"]

        with h5py.File(self.file_name, "r") as data:
            chunk_sizes = data["chunks"][()]

        start_idx = 0
        tmp_indicies = []
        for chunk_size in chunk_sizes:
            if chunk_size >= seq_len:
                tmp_indicies.append(
                    torch.arange(start_idx, start_idx + chunk_size).unfold(
                        0, seq_len, 1
                    )
                )
            start_idx += chunk_size

        indicies = torch.cat(tmp_indicies).int().tolist()
        self.indicies = random.sample(indicies, len(indicies))

    def __getitem__(self, index):
        datapoint = {}
        with h5py.File(self.file_name, "r") as data:
            datapoint["p1_face"] = data["p1_face"][self.indicies[index]][
                :, self.face_indicies
            ]

            if self.p1_speech_history:
                datapoint["p1_speech"] = data["p1_speech"][self.indicies[index]][
                    :, : self.speech_dim
                ]

            if self.p2_speech_history:
                datapoint["p2_speech"] = data["p2_speech"][self.indicies[index]][
                    :, : self.speech_dim
                ]

            if self.p2_face_history or "val" in self.file_name.stem:
                datapoint["p2_face"] = data["p2_face"][self.indicies[index]][
                    :, self.face_indicies
                ]

            if self.use_frame_nb:
                datapoint["frame_nb"] = data["frame_nb"][self.indicies[index]][0]

        return datapoint

    def __len__(self):
        return len(self.indicies)
