import sys

from glow_pytorch.glow.lets_face_it_glow import LetsFaceItGlow
from misc.shared import BASE_DIR, DATA_DIR, RANDOM_SEED
from pytorch_lightning import Trainer, seed_everything

seed_everything(RANDOM_SEED)


if __name__ == "__main__":
    
    model = LetsFaceItGlow.load_from_checkpoint(sys.argv[1], dataset_root=str(DATA_DIR))
    params = vars(model.hparams)
    if CONFIG["comet"]["api_key"]:
        from pytorch_lightning.loggers import CometLogger

        params["logger"] = CometLogger(
            api_key=CONFIG["comet"]["api_key"],
            project_name=CONFIG["comet"]["project_name"],
            experiment_name=sys.argv[1],
        )

    trainer = Trainer(**params)
    trainer.fit(model)
