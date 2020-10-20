import torch

from glow_pytorch.glow.utils import get_longest_history
import random


class TestMixin:
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

    def test_invertability(self, z_seq, loss, data):
        random_idx = random.randint(0, data["p1_face"].shape[0] - 1)
        start_ts = get_longest_history(self.hparams.Conditioning)
        reconstr_seq = []

        self.glow.init_rnn_hidden()
        backward_loss = 0
        frame_nb = None
        if self.hparams.Conditioning["use_frame_nb"]:
            frame_nb = data["frame_nb"].clone() + start_ts * 2

        for time_st, z_enc in enumerate(z_seq):
            condition = self.create_conditioning(
                data, start_ts + time_st, frame_nb, data["p1_face"]
            )

            reconstr, backward_objective = self.glow(
                z=z_enc, condition=condition, eps_std=1, reverse=True
            )
            backward_loss += torch.mean(
                self.loss(backward_objective, z_enc)
            )  # , x.size(1)

            if self.hparams.Conditioning["use_frame_nb"]:
                frame_nb += 2

            reconstr_seq.append(reconstr.detach())

        backward_loss = (backward_loss / len(z_seq)).unsqueeze(-1)

        error_percentage = (backward_loss + loss) / loss
        if self.hparams.Validation["render"]:
            seq = torch.stack(reconstr_seq, dim=1).type_as(data["p1_face"])[random_idx]
            gt = data["p1_face"][random_idx, -len(z_seq) :, :].detach()
            self.render("test_reconstr", gt, seq)

        return torch.abs(error_percentage)
