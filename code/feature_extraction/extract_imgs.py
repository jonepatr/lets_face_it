from tqdm import tqdm

from data_processing import img_utils
from misc.shared import DATA_DIR

for file_ in tqdm(list(DATA_DIR.glob("Sessions_50fps/*/*.mp4"))):
    new_folder = file_.parents[2] / "Sessions_50fps_imgs" / file_.parts[-2] / file_.stem
    img_utils.vid_to_img(file_, new_folder)
