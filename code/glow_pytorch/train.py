import os
from glow_pytorch.glow.lets_face_it_glow import LetsFaceItGlow
from glow_pytorch.glow.utils import get_hparams
from glow_pytorch.mimicry_data_module import MimicryDataModule
from glow_pytorch.mimicry_logger import MimicryLogger
from misc.shared import BASE_DIR, CONFIG, DATA_DIR, RANDOM_SEED
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser





seed_everything(RANDOM_SEED)


if __name__ == "__main__":
    hparams, conf_name = get_hparams()
    assert os.path.exists(
        hparams.dataset_root
    ), "Failed to find root dir `{}` of dataset.".format(hparams.dataset_root)

    model = LetsFaceItGlow(hparams)
    logger = None
    if CONFIG["comet"]["api_key"]:
        from pytorch_lightning.loggers import CometLogger

        logger = CometLogger(
            api_key=CONFIG["comet"]["api_key"],
            project_name=CONFIG["comet"]["project_name"],
        )
    
    callbacks = [MimicryLogger()]

    mimicry_data_module = MimicryDataModule(hparams)

    trainer = Trainer.from_argparse_args(hparams, logger=logger, callbacks=callbacks)
    trainer.fit(model, mimicry_data_module)
