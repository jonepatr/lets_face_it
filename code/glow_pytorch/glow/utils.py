import os
import numpy as np

from argparse import ArgumentParser, Namespace
from jsmin import jsmin
import json
import yaml
from pytorch_lightning import Trainer
from misc.shared import DATA_DIR




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
        hparams_json = yaml.load(open(override_params.hparams_file))
    hparams_json["dataset_root"] = str(DATA_DIR)

    params = vars(default_params)
    params.update(hparams_json)
    params.update(vars(override_params))

    hparams = Namespace(**params)

    return hparams, conf_name

def face_dim(data_hparams):
    return (
        data_hparams["expression_dim"]
        + data_hparams["jaw_dim"]
        + data_hparams["neck_dim"]
        + data_hparams["expression_delta_dim"]
        + data_hparams["jaw_delta_dim"]
        + data_hparams["neck_delta_dim"]
    )


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
