from pathlib import Path

from tqdm import tqdm

from data_processing.voca_processor import voca
from misc.shared import BASE_DIR, DATA_DIR
from misc.utils import replace_part


for dir_ in tqdm(list(DATA_DIR.glob("Sessions_wav/*/*"))):
    voca_dir = replace_part(dir_, "Sessions_wav", "Sessions_50fps_voca")
    ringnet_dir = replace_part(dir_, "Sessions_wav", "Sessions_50fps_ringnet")
    img_dir = replace_part(dir_, "Sessions_wav", "Sessions_50fps_imgs")

    voca(
        sorted(dir_.glob("*.wav")),
        ringnet_dir / "neutral_mesh.ply",
        voca_dir,
        len(list(img_dir.glob("*.jpg"))),
    )
