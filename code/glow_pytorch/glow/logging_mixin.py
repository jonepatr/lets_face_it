import io
import json
from pathlib import Path
from threading import Thread

import h5py
import numpy as np
import requests
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from glow_pytorch.glow.models import FlowStep
from glow_pytorch.glow.modules import ActNorm2d, InvertibleConv1x1
from misc.utils import get_face_indicies


class LoggingMixin:
    def log_scales(self):
        if not self.logger:
            return

        def log_histogram(x, name):
            if isinstance(self.logger, TensorBoardLogger):
                self.logger.experiment.add_histogram(name, x, self.global_step)
            else:
                self.logger.experiment.log_histogram_3d(x, name, self.global_step)

        for name, x in self.named_modules():
            if isinstance(x, ActNorm2d):
                log_histogram(torch.exp(x.logs), f"ActNorm/{name}")
            elif isinstance(x, FlowStep):
                log_histogram(x.scale, f"FlowStepScale/{name}")
            elif (
                isinstance(x, InvertibleConv1x1) and self.hparams.Glow["LU_decomposed"]
            ):
                log_histogram(torch.exp(x.log_s), f"InvertibleConv1x1_exp_log_s/{name}")

    def render_results(self, predicted_seq, batch, idx, name_prefix=""):
        output_seq_len = predicted_seq.shape[1]

        device = batch["p1_face"].device.index
        if batch.get("p2_face") is not None:
            p2_face = batch["p2_face"][idx]
            self.render(
                f"video/{device}", p2_face[-output_seq_len:], predicted_seq[idx],
            )
            self.render(
                f"gt_video/{device}",
                p2_face[-output_seq_len:],
                batch["p1_face"][idx, -output_seq_len:],
            )
        else:
            self.render(f"{name_prefix}video/{device}", predicted_seq[idx])
            self.render(
                f"{name_prefix}gt_video/{device}",
                batch["p1_face"][idx, -output_seq_len:],
            )

    def de_standardize(self, seq):
        face_indicies = get_face_indicies(
            self.hparams.Data["expression_dim"],
            self.hparams.Data["jaw_dim"],
            self.hparams.Data["neck_dim"],
        )

        with h5py.File(
            Path(self.hparams.dataset_root) / self.hparams.Train["data_file_name"], "r"
        ) as data:
            face_means = torch.tensor(
                data["standardization"]["face"]["means"][face_indicies]
            ).type_as(seq)
            face_stds = torch.tensor(
                data["standardization"]["face"]["stds"][face_indicies]
            ).type_as(seq)

        return seq * face_stds + face_means

    def render(self, name, sequence, sequence2=None):
        if isinstance(self.logger, TensorBoardLogger):
            return
        
        render_seq = [self.de_standardize(sequence)]
        if sequence2 is not None:
            render_seq.append(self.de_standardize(sequence2))

        # step = self.global_step

        file_name = f"{self.current_epoch}_{self.global_step}_{name}"

        def recvr(video):
            # self.logger.experiment.add_video(name, video, step, 25)
            self.logger.experiment.log_html(
                f"{file_name}<br><video src='{video}' width=640 controls></video> <br><br>"
            )

        file_path = f"{self.logger.project_name}/{self.logger.version}/{file_name}.mp4"
        
        self.async_render_file(render_seq, file_path, recvr)

    def async_render_file(self, render_seq, file_name, cb):
        def recvr():
            memfile = io.BytesIO()
            np.save(memfile, render_seq)
            memfile.seek(0)

            serialized = json.dumps(
                {
                    "seqs": memfile.read().decode("latin-1"),
                    "file_name": file_name,
                    "exp_dim": self.hparams.Data["expression_dim"],
                    "jaw_dim": self.hparams.Data["jaw_dim"],
                    "neck_dim": self.hparams.Data["neck_dim"],
                }
            )
            try:
                resp = requests.post(
                    "http://IP.IP.IP.IP:8000/render", data=serialized, timeout=600
                )
                resp.raise_for_status()
                cb(resp.json()["url"])
            except requests.exceptions.HTTPError:
                print("render request: failed on the server..")
            except requests.exceptions.Timeout:
                print("render request: timed out")
            except requests.exceptions.ConnectionError:
                print("render request: connection error")

        Thread(target=recvr, daemon=True).start()
