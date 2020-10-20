import h5py
from pathlib import Path
import torch
from glow_pytorch.glow_mimicry_dataset import GlowMimicryDataset
from torch.utils.data.dataloader import DataLoader


class DataMixin:
    def create_conditioning(self, data, time_st, frame_nb, prev_p1_faces):
        p1_prev_face_history = self.hparams.Conditioning["p1_face"]["history"]

        output = {
            "prev_p1_face": prev_p1_faces[:, time_st - p1_prev_face_history : time_st]
        }

        for modality in ["p1_speech", "p2_speech", "p2_face"]:
            history = self.hparams.Conditioning[modality]["history"]
            if history:
                output[modality] = data[modality][
                    :, (time_st - history) + 1 : time_st + 1
                ]

        if self.hparams.Conditioning["use_frame_nb"]:
            output["frame_nb"] = frame_nb

        return self.feature_encoder(output)

    def derange_batch(self, batch_data, modalities, shuffle_time=False):
        # Shuffle conditioning info
        batch_size = batch_data["p1_face"].size(0)
        permutation = torch.randperm(batch_size)

        mixed_up_batch = {}
        for modality in ["p1_face", "p2_face", "p1_speech", "p2_speech"]:
            if modality in modalities:
                mixed_up_batch[modality] = batch_data[modality][permutation]
                if shuffle_time:
                    t_perm = torch.randperm(batch_data[modality].size(1))
                    mixed_up_batch[modality] = mixed_up_batch[modality][:, t_perm]
            elif batch_data.get(modality) is not None:
                mixed_up_batch[modality] = batch_data[modality]

        return mixed_up_batch

    def _data_loader(self, file_name, shuffle=True, seq_len=25):

        dataset = GlowMimicryDataset(
            Path(self.hparams.dataset_root) / file_name,
            seq_len=seq_len,
            data_hparams=self.hparams.Data,
            conditioning_hparams=self.hparams.Conditioning,
        )

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_dataloader_workers,
            shuffle=shuffle,
            drop_last=False,
        )

    def train_dataloader(self):
        return self._data_loader(
            self.hparams.Train["data_file_name"], seq_len=self.hparams.Train["seq_len"],
        )

    def val_dataloader(self):
        return self._data_loader(
            self.hparams.Validation["data_file_name"],
            shuffle=False,
            seq_len=self.hparams.Validation["seq_len"],
        )

    def test_dataloader(self):
        return self._data_loader(
            self.hparams.Test["data_file_name"],
            shuffle=False,
            seq_len=self.hparams.Test["seq_len"],
        )
