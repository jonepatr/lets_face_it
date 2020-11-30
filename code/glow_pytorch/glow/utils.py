import json
import os
from argparse import ArgumentParser, Namespace

import torch
import yaml
from jsmin import jsmin
from misc.shared import DATA_DIR
from pytorch_lightning import Trainer
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR, StepLR


def get_hparams():
    parser = ArgumentParser()
    parser.add_argument("hparams_file")
    parser = Trainer.add_argparse_args(parser)
    default_params = parser.parse_args()

    parser2 = ArgumentParser()
    parser2.add_argument("hparams_file")
    override_params, unknown = parser2.parse_known_args()

    conf_name = os.path.basename(override_params.hparams_file)
    if override_params.hparams_file.endswith(".json"):
        hparams_json = json.loads(jsmin(open(override_params.hparams_file).read()))
    elif override_params.hparams_file.endswith(".yaml"):
        hparams_json = yaml.load(
            open(override_params.hparams_file), Loader=yaml.FullLoader
        )
    hparams_json["dataset_root"] = str(DATA_DIR)

    if not hparams_json["Glow"].get("rnn_type"):
        hparams_json["Glow"]["rnn_type"] = "gru"

    params = vars(default_params)
    params.update(hparams_json)
    params.update(vars(override_params))

    hparams = Namespace(**params)

    return hparams, conf_name


def get_longest_history(cond_params):
    return max(
        cond_params["p1_face"]["history"],
        cond_params["p1_speech"]["history"],
        cond_params["p2_speech"]["history"],
        cond_params["p2_face"]["history"],
    )


def calc_jerk(x):
    x = x.cpu()
    deriv = x[:, 1:] - x[:, :-1]
    acc = deriv[:, 1:] - deriv[:, :-1]
    jerk = acc[:, 1:] - acc[:, :-1]
    return jerk.abs().mean()


def lambda1(val):
    return lambda epoch: epoch // val


def get_scheduler(sched_params, optimizer):
    sched_name = sched_params["name"]

    if not sched_name:
        return optimizer

    sched_args = sched_params["args"][sched_name]

    if sched_name == "step":
        scheduler = StepLR(optimizer, **sched_args)
    elif sched_name == "multiplicative":
        scheduler = MultiplicativeLR(optimizer, lr_lambda=[lambda1(sched_args["val"])])
    elif sched_name == "lambda":
        scheduler = LambdaLR(optimizer, lr_lambda=[lambda1(sched_args["val"])])
    else:
        raise NotImplementedError("Unimplemented Scheduler!")

    return [scheduler]


def derange_batch(batch_data, modalities, shuffle_time=False):
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


def get_mismatched_modalities(hparams):
    p2_face_history = hparams.Conditioning["p2_face"]["history"]
    p2_speech_history = hparams.Conditioning["p2_speech"]["history"]

    modalities = []
    if p2_face_history > 0:
        modalities.append("p2_face")
    if p2_speech_history > 0:
        modalities.append("p2_speech")
    name = "p2" if len(modalities) == 2 else modalities[0]
    return modalities, name


def test_params(hparams):
    train_seq_len = hparams.Train["seq_len"]
    val_seq_len = hparams.Validation["seq_len"]
    for history in ["p1_face", "p2_face", "p1_speech", "p2_speech"]:
        his = hparams.Conditioning[history]["history"] + 1
        assert his < train_seq_len, f"{his} > {train_seq_len}"
        assert his < val_seq_len, f"{his} > {val_seq_len}"
