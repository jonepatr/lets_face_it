import io
import json
import random
from pathlib import Path
from threading import Thread

import h5py
import numpy as np
import requests
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger

from glow_pytorch.glow import calc_jerk, get_longest_history
from glow_pytorch.glow.lets_face_it_glow import derange_batch
from glow_pytorch.glow.models import FlowStep
from glow_pytorch.glow.modules import ActNorm2d, InvertibleConv1x1


class MimicryLogger(Callback):
    def render_results(self, predicted_seq, batch, idx, pl_module, name_prefix=""):
        output_seq_len = predicted_seq.shape[1]

        device = batch["p1_face"].device.index
        if batch.get("p2_face") is not None:
            p2_face = batch["p2_face"][idx]
            self.render(
                f"video/{device}",
                p2_face[-output_seq_len:],
                predicted_seq[idx],
                pl_module=pl_module,
            )
            self.render(
                f"gt_video/{device}",
                p2_face[-output_seq_len:],
                batch["p1_face"][idx, -output_seq_len:],
                pl_module=pl_module,
            )
        else:
            self.render(
                f"{name_prefix}video/{device}", predicted_seq[idx], pl_module=pl_module
            )
            self.render(
                f"{name_prefix}gt_video/{device}",
                batch["p1_face"][idx, -output_seq_len:],
                pl_module=pl_module,
            )

    def de_standardize(self, seq, hparams):
        data_file = Path(hparams.dataset_root) / hparams.Data["file_name"]
        with h5py.File(data_file, "r") as data:

            def get_standardization(type_):
                exp = data[type_]["flame_expression"][: hparams.Data["expression_dim"]]
                jaw = data[type_]["flame_jaw"]
                neck = data[type_]["flame_neck"]

                return np.concatenate([exp, jaw, neck])

            face_means = torch.tensor(get_standardization("means")).type_as(seq)
            face_stds = torch.tensor(get_standardization("stds")).type_as(seq)

        return seq * face_stds + face_means

    def render(self, name, sequence, sequence2=None, pl_module=None):
        if isinstance(pl_module.logger, TensorBoardLogger):
            return

        render_seq = [self.de_standardize(sequence, pl_module.hparams)]
        if sequence2 is not None:
            render_seq.append(self.de_standardize(sequence2, pl_module.hparams))

        file_name = f"{pl_module.current_epoch}_{pl_module.global_step}_{name}"

        def recvr(video):
            pl_module.logger.experiment.log_html(
                f"{file_name}<br><video src='{video}' width=640 controls></video> <br><br>"
            )

        file_path = (
            f"{pl_module.logger.name}/{pl_module.logger.version}/{file_name}.mp4"
        )

        self.async_render_file(render_seq, file_path, recvr, pl_module.hparams)

    def async_render_file(self, render_seq, file_name, cb, hparams):
        def recvr():
            def byteify(x):
                memfile = io.BytesIO()
                np.save(memfile, x.detach().cpu().numpy())
                memfile.seek(0)
                return memfile.read().decode("latin-1")

            def get_face(x):
                return {
                    "expression": byteify(x[:, :50]),
                    "pose": byteify(torch.zeros((x.shape[0], 12))),
                    "shape": byteify(torch.zeros((x.shape[0], 300))),
                    "rotation": byteify(torch.zeros((x.shape[0], 3))),
                }

            serialized = json.dumps(
                {
                    "seqs": [get_face(render_seq[0]), get_face(render_seq[1])],
                    "file_name": file_name,
                    "fps": 25,
                }
            )

            try:
                resp = requests.post(
                    "http://localhost:8000/render", data=serialized, timeout=600
                )
                resp.raise_for_status()
                print(resp.json())
                cb(resp.json()["url"])
            except requests.exceptions.HTTPError:
                print("render request: failed on the server..")
            except requests.exceptions.Timeout:
                print("render request: timed out")
            except requests.exceptions.ConnectionError:
                print("render request: connection error")

        Thread(target=recvr, daemon=True).start()

    def log_scales(self, pl_module):
        if not pl_module.logger:
            return

        def log_histogram(x, name):
            if isinstance(pl_module.logger, TensorBoardLogger):
                pl_module.logger.experiment.add_histogram(
                    name, x.detach().cpu(), pl_module.global_step
                )
            else:
                pl_module.logger.experiment.log_histogram_3d(
                    x.detach().cpu(), name=name, step=pl_module.global_step
                )

        for name, x in pl_module.named_modules():
            if isinstance(x, ActNorm2d):
                log_histogram(torch.exp(x.logs), f"ActNorm/{name}")
            elif isinstance(x, FlowStep):
                log_histogram(x.scale, f"FlowStepScale/{name}")
            elif (
                isinstance(x, InvertibleConv1x1)
                and pl_module.hparams.Glow["LU_decomposed"]
            ):
                log_histogram(
                    torch.exp(x.log_s),
                    f"InvertibleConv1x1_exp_log_s/{name}",
                )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        output = {}
        if batch_idx == 0:  #  and self.global_step > 0
            new_batch = {x: y.type_as(outputs) for x, y in batch.items()}
            z_seq, loss, _ = pl_module.seq_glow(new_batch)
            output["jerk"] = {}
            idx = random.randint(0, batch["p1_face"].shape[0] - 1)
            if pl_module.hparams.Validation["inference"]:
                seq_len = pl_module.hparams.Validation["seq_len"]
                cond_data = {
                    "p1_face": new_batch["p1_face"][
                        :, : get_longest_history(pl_module.hparams.Conditioning)
                    ],
                    "p2_face": new_batch.get("p2_face"),
                    "p1_speech": new_batch.get("p1_speech"),
                    "p2_speech": new_batch.get("p2_speech"),
                }
                predicted_seq = pl_module.seq_glow.inference(seq_len, data=cond_data)

                gt_mean_jerk = calc_jerk(
                    new_batch["p1_face"][:, -predicted_seq.shape[1] :]
                )
                generated_mean_jerk = calc_jerk(predicted_seq)

                pl_module.log("jerk/gt_mean", gt_mean_jerk)
                pl_module.log("jerk/generated_mean", generated_mean_jerk)
                pl_module.log(
                    "jerk/generated_mean_ratio", generated_mean_jerk / gt_mean_jerk
                )

                idx = random.randint(0, cond_data["p1_face"].shape[0] - 1)
                if pl_module.hparams.Validation["render"]:
                    self.render_results(predicted_seq, new_batch, idx, pl_module)

            if pl_module.hparams.Validation["check_invertion"]:
                # Test if the Flow works correctly
                det_check = self.test_invertability(z_seq, loss, new_batch, pl_module)
                pl_module.log("reconstruction/error_percentage", det_check)

            if pl_module.hparams.Validation["scale_logging"]:
                self.log_scales(pl_module)

            # Test if the Flow is listening to other modalities
            if pl_module.hparams.Validation["wrong_context_test"]:
                mismatch = pl_module.hparams.Mismatch
                pl_module.log(f"mismatched_nll/actual_nll", loss)

                for key, modalities in mismatch["shuffle_batch"].items():
                    if all(
                        [
                            pl_module.hparams.Conditioning[x]["history"] > 0
                            for x in modalities
                        ]
                    ):
                        deranged_batch = derange_batch(new_batch, modalities)
                        _, missaligned_nll, _ = pl_module.seq_glow(deranged_batch)

                        pl_module.log(
                            f"mismatched_nll/shuffle_batch_{key}", missaligned_nll
                        )
                        pl_module.log(
                            f"mismatched_nll_ratios/shuffle_batch_{key}",
                            loss - missaligned_nll,
                        )

                for key, modalities in mismatch["shuffle_time"].items():
                    if all(
                        [
                            pl_module.hparams.Conditioning[x]["history"] > 0
                            for x in modalities
                        ]
                    ):
                        deranged_batch = derange_batch(
                            new_batch, modalities, shuffle_time=True
                        )
                        _, shuffled_nll, _ = pl_module.seq_glow(deranged_batch)
                        pl_module.log(
                            f"mismatched_nll/shuffle_time_{key}", shuffled_nll
                        )
                        pl_module.log(
                            f"mismatched_nll_ratios/shuffle_time_{key}",
                            loss - shuffled_nll,
                        )


    def test_invertability(self, z_seq, loss, data, pl_module):
        reconstr_seq, backward_loss = pl_module.seq_glow.invert(z_seq, data)

        random_idx = random.randint(0, data["p1_face"].shape[0] - 1)
        error_percentage = (backward_loss + loss) / loss
        if pl_module.hparams.Validation["render"]:
            seq = torch.stack(reconstr_seq, dim=1).type_as(data["p1_face"])[random_idx]
            gt = data["p1_face"][random_idx, -len(z_seq) :, :].detach()
            self.render("test_reconstr", gt, seq, pl_module=pl_module)

        return torch.abs(error_percentage)
