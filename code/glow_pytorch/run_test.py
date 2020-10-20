from glow_pytorch.glow.lets_face_it_glow import LetsFaceItGlow
from misc.shared import DATA_DIR, BASE_DIR, RANDOM_SEED
from pytorch_lightning import Trainer, seed_everything
import torch


seed_everything(RANDOM_SEED)


def test(model_path, file_name):
    model = LetsFaceItGlow.load_from_checkpoint(
        model_path,
        dataset_root=str(DATA_DIR),
        test={"seq_len": 100, "data_file_name": "test.hdf5"},
    )
    model.hparams.batch_size = 33000
    trainer = Trainer(gpus=1, single_gpu=True, num_sanity_val_steps=0)
    trainer.test(model)
    torch.save(trainer.callback_metrics["results"], file_name)


if __name__ == "__main__":
    test(BASE_DIR / "final_model.ckpt", BASE_DIR / "results/test_results.pt")
