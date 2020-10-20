import os
from glow_pytorch.glow.lets_face_it_glow import LetsFaceItGlow
from glow_pytorch.glow.utils import get_hparams
from misc.shared import BASE_DIR, CONFIG, DATA_DIR, RANDOM_SEED
from pytorch_lightning import Trainer, seed_everything

seed_everything(RANDOM_SEED)


if __name__ == "__main__":
    hparams, conf_name = get_hparams()
    assert os.path.exists(
        hparams.dataset_root
    ), "Failed to find root dir `{}` of dataset.".format(hparams.dataset_root)

    hparams.num_dataloader_workers = 0
    hparams.gpus = 1

    model = LetsFaceItGlow(hparams)
    trainer_params = vars(hparams).copy()
    if CONFIG["comet"]["api_key"]:
        from pytorch_lightning.loggers import CometLogger

        trainer_params["logger"] = CometLogger(
            api_key=CONFIG["comet"]["api_key"],
            project_name=CONFIG["comet"]["project_name"],
        )
    
    trainer = Trainer(**trainer_params)
    trainer.fit(model)
