from tarfile import TarFile

import numpy as np
from tqdm import tqdm

from misc.shared import DATA_DIR
from misc.utils import replace_part

for dir_ in tqdm(list(DATA_DIR.glob("Sessions_50fps/*/*"))):
    openface_landmarks = replace_part(dir_, "Sessions_50fps", "Sessions_50fps_openface_51_landmarks").with_suffix("")
    
    tar_file = TarFile(replace_part(dir_, "Sessions_50fps", "Sessions_50fps_openface").with_suffix(".tar"))
    openface_data = {}
    openface_landmarks.mkdir(parents=True, exist_ok=True)
    d = tar_file.extractfile([x for x in tar_file.getmembers() if x.path.endswith(".csv")][0]).readlines()

    for i, line in enumerate(d):
        if i == 0:
            continue
        split_line = line.decode('utf-8').split(",")
        
        
        npy_file = (openface_landmarks / split_line[0].zfill(5)).with_suffix(".npy")
        data = np.array(list(float(x) for x in split_line[299:435])).reshape(2, -1).T[17:]

        np.save(npy_file, data)
