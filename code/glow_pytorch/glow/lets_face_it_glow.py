import random
from glow_pytorch.glow.utils import (
    derange_batch,
    get_mismatched_modalities,
    get_scheduler,
    test_params,
)

import numpy as np
import optuna
import torch
from glow_pytorch.glow import get_longest_history
from glow_pytorch.glow.models import SeqGlow
from pytorch_lightning import LightningModule
from torch.optim import SGD, Adam, RMSprop


class LetsFaceItGlow(LightningModule):
    def __init__(self, hparams, dataset_root=None, test=None):
        super().__init__()

        test_params(hparams)
        if dataset_root is not None:
            hparams.dataset_root = dataset_root
        if test is not None:
            hparams.Test = test

        self.hparams = hparams
        self.register_buffer("last_missmatched_nll", torch.tensor(np.Inf))

        self.seq_glow = SeqGlow(self.hparams)

        if self.hparams.Train["use_negative_nll_loss"]:
            modalities, nll_name = get_mismatched_modalities(self.hparams)

            self.missmatched_modalities = modalities
            self.missmatched_nll_name = nll_name

    def training_step(self, batch, batch_idx):
        if (
            self.hparams.Train["use_negative_nll_loss"]
            and self.last_missmatched_nll > 0
            and random.random() < 0.1
            and self.missmatched_modalities
        ):
            deranged_batch = derange_batch(batch, self.missmatched_modalities)
            _, loss, _ = self.seq_glow(deranged_batch)
            self.log("Loss/missmatched_nll", -loss)
            self.last_missmatched_nll = -loss
            loss *= -0.1
        else:
            _, loss, _ = self.seq_glow(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss, _ = self.seq_glow(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer_name = self.hparams.Optim["name"]

        # Define optimizer
        optimizers = {"adam": Adam, "sgd": SGD, "rmsprop": RMSprop}
        optimizer = optimizers[optimizer_name](
            self.parameters(),
            lr=self.hparams.lr,
            **self.hparams.Optim["args"][optimizer_name],
        )

        return [optimizer], get_scheduler(self.hparams.Optim["Schedule"], optimizer)

    # learning rate warm-up
    # def optimizer_step(
    #     self,
    #     current_epoch,
    #     batch_nb,
    #     optimizer,
    #     optimizer_idx,
    #     second_order_closure=None,
    #     on_tpu=False,
    #     using_native_amp=False,
    #     using_lbfgs=False,
    # ):
    #     lr = self.hparams.lr
    #     # warm up lr
    #     warm_up = self.hparams.Optim["Schedule"]["warm_up"]
    #     if self.trainer.global_step < warm_up:
    #         lr_scale = min(1.0, float(self.trainer.global_step + 1) / warm_up)
    #         lr *= lr_scale
    #         for pg in optimizer.param_groups:
    #             pg["lr"] = lr
    #     for pg in optimizer.param_groups:
    #         self.log("learning_rate", pg["lr"])

    #     # update params
    #     optimizer.step()
    #     optimizer.zero_grad()

    def test_step(self, batch, batch_idx):
        _, loss, losses = self(batch, test=True)
        output = {"test_loss": loss, "test_losses": losses}

        seq_len = self.hparams.Test["seq_len"]
        cond_data = {
            "p1_face": torch.zeros_like(
                batch["p1_face"][:, : get_longest_history(self.hparams.Conditioning)]
            ),
            "p2_face": batch.get("p2_face"),
            "p1_speech": batch.get("p1_speech"),
            "p2_speech": batch.get("p2_speech"),
        }
        predicted_seq = self.inference(seq_len, data=cond_data)
        output["predicted_prop_seq"] = predicted_seq.cpu().detach()
        gt_seq = batch["p1_face"][:, -predicted_seq.shape[1] :]
        output["gt_seq"] = gt_seq.cpu().detach()
        for modality in ["p2_face", "p2_speech", "p1_speech"]:
            if self.hparams.Conditioning[modality]["history"] > 0:

                deranged_batch = self.derange_batch(batch, [modality])
                _, missaligned_nll, misaligned_losses = self(deranged_batch, test=True)
                output[f"nll_mismatched_{modality}"] = missaligned_nll.cpu().detach()
                output[f"losses_mismatched_{modality}"] = misaligned_losses

                cond_data = {
                    "p1_face": torch.zeros_like(
                        deranged_batch["p1_face"][
                            :, : get_longest_history(self.hparams.Conditioning)
                        ]
                    ),
                    "p2_face": deranged_batch.get("p2_face"),
                    "p1_speech": deranged_batch.get("p1_speech"),
                    "p2_speech": deranged_batch.get("p2_speech"),
                }

                predicted_seq = self.inference(seq_len, data=cond_data)
                output[
                    f"predicted_mismatch_{modality}_seq"
                ] = predicted_seq.cpu().detach()

        return output

    def test_epoch_end(self, outputs):
        return {"results": outputs}
