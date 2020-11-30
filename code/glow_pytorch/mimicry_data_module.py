import random
from pathlib import Path

import h5py
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class MimicryDataset(Dataset):
    def __init__(
        self,
        file_name,
        data_type,
        data_hparams=None,
        conditioning_hparams=None,
        seq_len=None,
    ):
        super().__init__()
        self.file_name = file_name
        self.data_type = data_type
        self.expression_dim = data_hparams["expression_dim"]

        self.speech_dim = data_hparams["speech_dim"]
        self.p1_speech_history = conditioning_hparams["p1_speech"]["history"]
        self.p2_speech_history = conditioning_hparams["p2_speech"]["history"]
        self.p2_face_history = conditioning_hparams["p2_face"]["history"]

        self.use_frame_nb = conditioning_hparams["use_frame_nb"]

        chunk_sizes = h5py.File(self.file_name, "r")[self.data_type]["prosody"]

        tmp_indicies = []

        for key, chunk in chunk_sizes.items():
            if len(chunk["agent"]) >= seq_len:
                for seq in torch.arange(len(chunk["agent"])).unfold(0, seq_len, 1):
                    tmp_indicies.append((key, seq.int().tolist()))

        self.indicies = random.sample(tmp_indicies, len(tmp_indicies))

    def __getitem__(self, index):
        datapoint = {}
        indicies = self.indicies[index]
        with h5py.File(self.file_name, "r") as data:

            def get_data_item(kind, who):
                return data[self.data_type][kind][indicies[0]][who][indicies[1]]

            def get_flame_face(who):
                expression = get_data_item("flame_expression", who)[
                    :, : self.expression_dim
                ]
                jaw = get_data_item("flame_jaw", who)
                neck = get_data_item("flame_neck", who)
                return torch.from_numpy(
                    np.concatenate([expression, jaw, neck], axis=1)
                ).float()

            def get_speech(who):
                mfcc = get_data_item("mfcc", who)
                prosody = get_data_item("prosody", who)
                return torch.from_numpy(np.concatenate([mfcc, prosody], axis=1)).float()

            datapoint["p1_face"] = get_flame_face("agent")

            if self.p1_speech_history:
                datapoint["p1_speech"] = get_speech("agent")

            if self.p2_speech_history:
                datapoint["p2_speech"] = get_speech("interlocutor")

            if self.p2_face_history:
                datapoint["p2_face"] = get_flame_face("interlocutor")

        return datapoint

    def __len__(self):
        return len(self.indicies)


class MimicryDataModule(LightningDataModule):
    def __init__(self, hparams, workers=8):
        super().__init__()
        self.hparams = hparams
        self.workers = workers
        self.file_name = Path(hparams.dataset_root) / self.hparams.Data["file_name"]

    def _data_loader(self, data_type, shuffle=True, seq_len=25):

        dataset = MimicryDataset(
            self.file_name,
            data_type,
            seq_len=seq_len,
            data_hparams=self.hparams.Data,
            conditioning_hparams=self.hparams.Conditioning,
        )

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.workers,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._data_loader(
            "train",
            seq_len=self.hparams.Train["seq_len"],
        )

    def val_dataloader(self):
        return self._data_loader(
            "val",
            shuffle=False,
            seq_len=self.hparams.Validation["seq_len"],
        )

    def test_dataloader(self):
        return self._data_loader(
            "test",
            shuffle=False,
            seq_len=self.hparams.Test["seq_len"],
        )
