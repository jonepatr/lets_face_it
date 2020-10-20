import random
from functools import reduce

import numpy as np
import optuna
import torch
from pytorch_lightning import LightningModule
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR, StepLR
from torch.utils.data import DataLoader
from glow_pytorch.glow.modules import GaussianDiag


from glow_pytorch.glow import (
    DataMixin,
    FeatureEncoder,
    Glow,
    calc_jerk,
    get_longest_history,
    LoggingMixin,
    TestMixin,
)


class LetsFaceItGlow(DataMixin, LoggingMixin, TestMixin, LightningModule):
    def __init__(self, hparams, dataset_root=None, test=None):
        super().__init__()

        self.test_params(hparams)
        if dataset_root is not None:
            hparams.dataset_root = dataset_root
        if test is not None:
            hparams.Test = test

        if not hparams.Glow.get("rnn_type"):
            hparams.Glow["rnn_type"] = "gru"

        self.hparams = hparams
        self.best_jerk = torch.tensor(np.Inf)
        self.last_missmatched_nll = torch.tensor(np.Inf)
        self.feature_encoder = FeatureEncoder(
            self.hparams.Conditioning, self.hparams.Data
        )

        self.glow = Glow(hparams, self.feature_encoder.dim)

        if self.hparams.Train["use_negative_nll_loss"]:
            p2_face_history = self.hparams.Conditioning["p2_face"]["history"]
            p2_speech_history = self.hparams.Conditioning["p2_speech"]["history"]

            if p2_face_history > 0 and p2_speech_history > 0:
                self.missmatched_modalities = ["p2_face", "p2_speech"]
                self.missmatched_nll_name = "p2"
            elif p2_face_history > 0:
                self.missmatched_modalities = ["p2_face"]
                self.missmatched_nll_name = "p2_face"
            elif p2_speech_history > 0:
                self.missmatched_modalities = ["p2_speech"]
                self.missmatched_nll_name = "p2_speech"
            else:
                self.missmatched_modalities = None

    def test_params(self, hparams):
        train_seq_len = hparams.Train["seq_len"]
        val_seq_len = hparams.Validation["seq_len"]
        for history in ["p1_face", "p2_face", "p1_speech", "p2_speech"]:
            his = hparams.Conditioning[history]["history"] + 1
            assert his < train_seq_len, f"{his} > {train_seq_len}"
            assert his < val_seq_len, f"{his} > {val_seq_len}"


    def inference(self, seq_len, data=None):
        self.glow.init_rnn_hidden()

        output_shape = torch.zeros_like(data["p1_face"][:, 0, :])
        frame_nb = None
        if self.hparams.Conditioning["use_frame_nb"]:
            frame_nb = torch.ones((data["p1_face"].shape[0], 1)).type_as(
                data["p1_face"]
            )

        prev_p1_faces = data["p1_face"]

        start_ts = get_longest_history(self.hparams.Conditioning)

        for time_st in range(start_ts, seq_len):
            condition = self.create_conditioning(data, time_st, frame_nb, prev_p1_faces)

            output, _ = self.glow(
                condition=condition,
                eps_std=self.hparams.Infer["eps"],
                reverse=True,
                output_shape=output_shape,
            )

            prev_p1_faces = torch.cat([prev_p1_faces, output.unsqueeze(1)], dim=1)

            if self.hparams.Conditioning["use_frame_nb"]:
                frame_nb += 2

        return prev_p1_faces[:, start_ts:]

    def forward(self, batch):
        self.glow.init_rnn_hidden()

        loss = 0
        start_ts = get_longest_history(self.hparams.Conditioning)

        frame_nb = None
        if self.hparams.Conditioning["use_frame_nb"]:
            frame_nb = batch["frame_nb"].clone() + start_ts * 2

        z_seq = []
        losses = []
        for time_st in range(start_ts, batch["p1_face"].shape[1]):
            curr_input = batch["p1_face"][:, time_st, :]
            condition = self.create_conditioning(
                batch, time_st, frame_nb, batch["p1_face"]
            )

            z_enc, objective = self.glow(x=curr_input, condition=condition)
            tmp_loss = self.loss(objective, z_enc)
            losses.append(tmp_loss.cpu().detach())
            loss += torch.mean(tmp_loss)

            if self.hparams.Conditioning["use_frame_nb"]:
                frame_nb += 2
            z_seq.append(z_enc.detach())

        return z_seq, (loss / len(z_seq)).unsqueeze(-1), losses

    def loss(self, objective, z):
        objective += GaussianDiag.logp_simplified(z)
        nll = (-objective) / float(np.log(2.0))
        return nll

    def training_step(self, batch, batch_idx):
        if (
            self.hparams.Train["use_negative_nll_loss"]
            and self.last_missmatched_nll > 0
            and random.random() < 0.1
            and self.missmatched_modalities
        ):
            deranged_batch = self.derange_batch(batch, self.missmatched_modalities)
            _, loss, _ = self(deranged_batch)

            tb_log = {"Loss/missmatched_nll": -loss}
            self.last_missmatched_nll = -loss
            loss *= -0.1
        else:
            _, loss, _ = self(batch)
            tb_log = {"Loss/train": loss}

            if self.hparams.optuna and self.global_step > 20 and loss > 0:
                message = f"Trial was pruned since loss > 0"
                raise optuna.exceptions.TrialPruned(message)

        return {"loss": loss, "log": tb_log}

    def validation_step(self, batch, batch_idx):
        z_seq, loss, _ = self(batch)
        if self.hparams.optuna and self.global_step > 20 and loss > 0:
            message = f"Trial was pruned since loss > 0"
            raise optuna.exceptions.TrialPruned(message)
        output = {"val_loss": loss}

        if batch_idx == 0:  #  and self.global_step > 0
            output["jerk"] = {}
            idx = random.randint(0, batch["p1_face"].shape[0] - 1)
            if self.hparams.Validation["inference"]:
                seq_len = self.hparams.Validation["seq_len"]
                cond_data = {
                    "p1_face": batch["p1_face"][
                        :, : get_longest_history(self.hparams.Conditioning)
                    ],
                    "p2_face": batch.get("p2_face"),
                    "p1_speech": batch.get("p1_speech"),
                    "p2_speech": batch.get("p2_speech"),
                }
                predicted_seq = self.inference(seq_len, data=cond_data)

                output["jerk"]["gt_mean"] = calc_jerk(
                    batch["p1_face"][:, -predicted_seq.shape[1] :]
                )
                output["jerk"]["generated_mean"] = calc_jerk(predicted_seq)

                idx = random.randint(0, cond_data["p1_face"].shape[0] - 1)
                if self.hparams.Validation["render"]:
                    self.render_results(predicted_seq, batch, idx)

            if self.hparams.Validation["check_invertion"]:
                # Test if the Flow works correctly
                output["det_check"] = self.test_invertability(z_seq, loss, batch)

            if self.hparams.Validation["scale_logging"]:
                self.log_scales()

            # Test if the Flow is listening to other modalities
            if self.hparams.Validation["wrong_context_test"]:
                mismatch = self.hparams.Mismatch
                output["mismatched_nll"] = {"actual_nll": loss}

                for key, modalities in mismatch["shuffle_batch"].items():
                    if all(
                        [
                            self.hparams.Conditioning[x]["history"] > 0
                            for x in modalities
                        ]
                    ):
                        deranged_batch = self.derange_batch(batch, modalities)
                        _, missaligned_nll, _ = self(deranged_batch)

                        output["mismatched_nll"][
                            f"shuffle_batch_{key}"
                        ] = missaligned_nll

                for key, modalities in mismatch["shuffle_time"].items():
                    if all(
                        [
                            self.hparams.Conditioning[x]["history"] > 0
                            for x in modalities
                        ]
                    ):
                        deranged_batch = self.derange_batch(
                            batch, modalities, shuffle_time=True
                        )
                        _, shuffled_nll, _ = self(deranged_batch)

                        output["mismatched_nll"][f"shuffle_time_{key}"] = shuffled_nll
        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tb_logs = {"Loss/val": avg_loss}
        save_loss = avg_loss

        mismatched_nll = [
            x["mismatched_nll"] for x in outputs if x.get("mismatched_nll")
        ]
        if mismatched_nll:
            keys = reduce(
                lambda x, y: x + y, [[y for y in x.keys()] for x in mismatched_nll]
            )
            actual_nll = torch.stack(
                [x["actual_nll"] for x in mismatched_nll if x.get("actual_nll")]
            ).mean()
            for key in keys:
                bad_nll = torch.stack(
                    [x[key] for x in mismatched_nll if x.get(key)]
                ).mean()
                tb_logs[f"mismatched_nll/{key}"] = bad_nll
                tb_logs[f"mismatched_nll_ratios/{key}"] = actual_nll - bad_nll

            if self.hparams.Train["use_negative_nll_loss"] and self.global_step > 0:
                self.last_missmatched_nll = -tb_logs[
                    f"mismatched_nll/shuffle_batch_{self.missmatched_nll_name}"
                ]

        det_check = [x["det_check"] for x in outputs if x.get("det_check") is not None]
        if det_check:
            avg_det_check = torch.stack(det_check).mean()
            tb_logs["reconstruction/error_percentage"] = avg_det_check

        jerk = [x["jerk"] for x in outputs if x.get("jerk")]
        if jerk:
            gt_jerk_mean = [x["gt_mean"] for x in jerk]
            if gt_jerk_mean:
                tb_logs[f"jerk/gt_mean"] = torch.stack(gt_jerk_mean).mean()

            generated_jerk_mean = [x["generated_mean"] for x in jerk]
            if generated_jerk_mean:
                tb_logs[f"jerk/generated_mean"] = torch.stack(
                    generated_jerk_mean
                ).mean()
                percentage = tb_logs[f"jerk/generated_mean"] / tb_logs[f"jerk/gt_mean"]
                tb_logs[f"jerk/generated_mean_ratio"] = percentage

            if (
                tb_logs[f"jerk/generated_mean"] > 5
                and self.hparams.optuna
                and self.global_step > 20
            ):
                message = f"Trial was pruned since jerk > 5"
                raise optuna.exceptions.TrialPruned(message)
            if tb_logs[f"jerk/generated_mean"] < self.best_jerk:
                self.best_jerk = tb_logs[f"jerk/generated_mean"]
            else:
                save_loss + torch.tensor(np.Inf)
        return {"save_loss": save_loss, "val_loss": avg_loss, "log": tb_logs}

    def configure_optimizers(self):
        lr_params = self.hparams.Optim
        optim_args = lr_params["args"][lr_params["name"]]
        optimizers = {"adam": Adam, "sgd": SGD, "rmsprop": RMSprop}
        # Define optimizer
        optimizer = optimizers[lr_params["name"]](
            self.parameters(), lr=self.hparams.lr, **optim_args
        )

        # Define Learning Rate Scheduling
        def lambda1(val):
            return lambda epoch: epoch // val

        sched_params = self.hparams.Optim["Schedule"]
        sched_name = sched_params["name"]
        if not sched_name:
            return optimizer

        sched_args = sched_params["args"][sched_name]

        if sched_name == "step":
            scheduler = StepLR(optimizer, **sched_args)
        elif sched_name == "multiplicative":
            scheduler = MultiplicativeLR(
                optimizer, lr_lambda=[lambda1(sched_args["val"])]
            )
        elif sched_name == "lambda":
            scheduler = LambdaLR(optimizer, lr_lambda=[lambda1(sched_args["val"])])
        else:
            raise NotImplementedError("Unimplemented Scheduler!")

        return [optimizer], [scheduler]

    # learning rate warm-up
    def optimizer_step(
        self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None
    ):
        lr = self.hparams.lr
        # warm up lr
        warm_up = self.hparams.Optim["Schedule"]["warm_up"]
        if self.trainer.global_step < warm_up:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / warm_up)
            lr *= lr_scale
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        for pg in optimizer.param_groups:
            self.logger.log_metrics({"learning_rate": pg["lr"]}, self.global_step)

        # update params
        optimizer.step()
        optimizer.zero_grad()
