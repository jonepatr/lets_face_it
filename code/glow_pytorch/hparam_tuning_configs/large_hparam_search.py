def hparam_options(hparams, trial):

    hparams.Glow["K"] = trial.suggest_categorical("K", [4, 8, 16, 32])

    hparams.Conditioning["cond_dim"] = trial.suggest_categorical(
        "cond_dim", [64, 128, 256, 512, 1024]
    )

    hparams.Optim["name"] = trial.suggest_categorical(
        "optim_name", ["adam", "sgd", "rmsprop"]
    )

    hparams.Optim["Schedule"]["name"] = trial.suggest_categorical(
        "Schedule_name", [None, "step"]
    )

    hparams.Optim["Schedule"]["args"]["step"]["gamma"] = trial.suggest_uniform(
        "Schedule_gamma", 0, 1
    )

    hparams.Optim["Schedule"]["args"]["step"]["step_size"] = trial.suggest_int(
        "Schedule_step_size", 1, 10
    )

    hparams.Train["use_negative_nll_loss"] = trial.suggest_categorical(
        "use_negative_nll_loss", [True, False]
    )

    hparams.Optim["Schedule"]["warm_up"] = trial.suggest_int("lr_warm_up", 0, 4000)

    hparams.Glow["hidden_channels"] = trial.suggest_categorical(
        "hidden_channels", [16, 32, 64, 128, 256, 512]
    )

    hparams.lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)

    hparams.Train["use_negative_nll_loss"] = trial.suggest_categorical(
        "use_negative_nll_loss", [True, False]
    )

    hparams.Train["seq_len"] = trial.suggest_int("seq_len", 20, 90)

    hparams.Data["expression_dim"] = trial.suggest_int("expression_dim", 5, 100)

    def enc_fixer(name, hist, hidden, dim):
        history = trial.suggest_categorical(f"{name}_history", hist)
        return {
            "dropout": trial.suggest_uniform(f"{name}_dropout", 0, 1),
            "enc": trial.suggest_categorical(f"{name}_enc", ["rnn", "mlp", "none"]),
            "history": history,
            "hidden_dim": trial.suggest_categorical(f"{name}_hidden_dim", hidden),
        }

    face_hist = [2, 4, 8, 16, 24]
    face_dim = hparams.Data["expression_dim"] + 6
    face_hidden = [128, 256, 512]

    speech_hist = [2, 4, 8, 16]
    speech_dim = 30
    speech_hidden = [64, 128, 256]
    hparams.Conditioning["p1_face"] = enc_fixer(
        "p1_face", face_hist, face_hidden, face_dim
    )
    hparams.Conditioning["p2_face"] = enc_fixer(
        "p2_face", face_hist, face_hidden, face_dim
    )
    hparams.Conditioning["p1_speech"] = enc_fixer(
        "p1_speech", speech_hist, speech_hidden, speech_dim
    )
    hparams.Conditioning["p2_speech"] = enc_fixer(
        "p2_speech", speech_hist, speech_hidden, speech_dim
    )

    return hparams
