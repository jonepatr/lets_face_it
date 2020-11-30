import json
import multiprocessing
import os
import shutil
import socket
from argparse import ArgumentParser, Namespace
from pprint import pprint
from glow_pytorch.glow.utils import calc_jerk, get_longest_history

import numpy as np
import optuna
import pytorch_lightning as pl
import torch
import yaml
from jsmin import jsmin
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer, seed_everything

from glow_pytorch.glow.lets_face_it_glow import LetsFaceItGlow
from glow_pytorch.hparam_tuning_configs import hparam_configs
from misc.shared import CONFIG, DATA_DIR, RANDOM_SEED
from misc.utils import get_training_name

seed_everything(RANDOM_SEED)


class FailedTrial(Exception):
    pass


class MyEarlyStopping(PyTorchLightningPruningCallback):
    def __init__(self, trial, monitor="val_loss", patience=2):
        super().__init__(trial, monitor=monitor)
        self.best_loss = torch.tensor(np.Inf)
        self.wait = 0
        self.patience = patience
        self.jerk_generated_means = []

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        if pl_module.global_step > 20 and outputs > 0:
            message = f"Trial was pruned since loss > 0"
            raise optuna.exceptions.TrialPruned(message)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.jerk_generated_means = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        seq_len = pl_module.hparams.Validation["seq_len"]
        new_batch = {x: y.type_as(outputs) for x, y in batch.items()}
        cond_data = {
            "p1_face": new_batch["p1_face"][
                :, : get_longest_history(pl_module.hparams.Conditioning)
            ],
            "p2_face": new_batch.get("p2_face"),
            "p1_speech": new_batch.get("p1_speech"),
            "p2_speech": new_batch.get("p2_speech"),
        }
        predicted_seq = pl_module.seq_glow.inference(seq_len, data=cond_data)
        self.jerk_generated_means.append(calc_jerk(predicted_seq))

        if pl_module.global_step > 20 and outputs > 0:
            message = f"Trial was pruned since loss > 0"
            raise optuna.exceptions.TrialPruned(message)

    def on_validation_epoch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        super().on_validation_epoch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        jerk_generated_mean = torch.stack(self.jerk_generated_means).mean()
        val_loss = torch.stack(outputs).mean()

        if jerk_generated_mean > 10 and pl_module.global_step > 20:
            message = f"Trial was pruned since jerk > 5"
            raise optuna.exceptions.TrialPruned(message)

        if val_loss is not None and val_loss > 0:
            message = f"Trial was pruned because val loss was too high {val_loss}."
            raise optuna.exceptions.TrialPruned(message)

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return True


parser = ArgumentParser()
parser.add_argument("hparams_file")
parser.add_argument("-n", type=int)
parser = Trainer.add_argparse_args(parser)
default_params = parser.parse_args()

parser2 = ArgumentParser()
parser2.add_argument("hparams_file")
parser2.add_argument("-n", type=int)
override_params, unknown = parser2.parse_known_args()

conf_name = (
    os.path.basename(override_params.hparams_file)
    .replace(".yaml", "")
    .replace(".json", "")
)


def prepare_hparams(trial):

    if override_params.hparams_file.endswith(".json"):
        hparams_json = json.loads(jsmin(open(override_params.hparams_file).read()))
    elif override_params.hparams_file.endswith(".yaml"):
        hparams_json = yaml.load(open(override_params.hparams_file))
    hparams_json["dataset_root"] = str(DATA_DIR)

    params = vars(default_params)
    params.update(hparams_json)
    params.update(vars(override_params))
    hparams = Namespace(**params)

    return hparam_configs[conf_name].hparam_options(hparams, trial)


def run(hparams, return_dict, trial, batch_size, current_date):

    log_path = os.path.join("logs", conf_name, f"{current_date}")
    if os.path.exists(log_path):
        shutil.rmtree(log_path)

    hparams.batch_size = batch_size

    trainer_params = vars(hparams).copy()
    trainer_params["checkpoint_callback"] = pl.callbacks.ModelCheckpoint(
        save_top_k=3, monitor="val_loss", mode="min"
    )

    if CONFIG["comet"]["api_key"]:
        from pytorch_lightning.loggers import CometLogger

        trainer_params["logger"] = CometLogger(
            api_key=CONFIG["comet"]["api_key"],
            project_name=CONFIG["comet"]["project_name"],
            experiment_name=conf_name,  # + current_date
        )

    trainer_params["early_stop_callback"] = MyEarlyStopping(trial, monitor="val_loss")

    trainer = Trainer(**trainer_params)
    model = LetsFaceItGlow(hparams)

    try:
        trainer.fit(model)
    except RuntimeError as e:
        if str(e).startswith("CUDA out of memory"):
            return_dict["OOM"] = True
        else:
            return_dict["error"] = e
    except (optuna.exceptions.TrialPruned, Exception) as e:
        return_dict["error"] = e

    for key, item in trainer.callback_metrics.items():
        return_dict[key] = float(item)


def objective(trial):
    current_date = get_training_name()

    manager = multiprocessing.Manager()

    hparams = prepare_hparams(trial)
    batch_size = hparams.batch_size

    trial.set_user_attr("version", current_date)
    trial.set_user_attr("host", socket.gethostname())
    trial.set_user_attr("GPU", os.environ.get("CUDA_VISIBLE_DEVICES"))

    pprint(vars(hparams))
    while batch_size > 0:
        print(f"trying with batch_size {batch_size}")

        return_dict = manager.dict()
        p = multiprocessing.Process(
            target=run,
            args=(hparams, return_dict, trial, batch_size, current_date),
        )
        p.start()
        p.join()
        print(return_dict)
        if return_dict.get("OOM"):
            new_batch_size = batch_size // 2
            if new_batch_size < 2:
                raise FailedTrial("batch size smaller than 2!")
            else:
                batch_size = new_batch_size
        elif return_dict.get("error"):
            raise return_dict.get("error")
        else:
            break
    trial.set_user_attr("batch_size", batch_size)

    for metric, val in return_dict.items():
        if metric != "val_loss":
            trial.set_user_attr(metric, float(val))

    return float(return_dict["val_loss"])


if __name__ == "__main__":
    conf_vars = {}
    if CONFIG["optuna"]["rdbs_storage"]:
        conf_vars["storage"] = optuna.storages.RDBStorage(
            url=CONFIG["optuna"]["rdbs_storage"],
        )

    study = optuna.create_study(
        **conf_vars,
        study_name=conf_name,
        direction="minimize",
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=override_params.n, catch=(FailedTrial,))

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
