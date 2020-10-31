from pathlib import Path
import tempfile
import ffmpeg
from misc.shared import DATASET_DIR, DATA_DIR
from tqdm import tqdm
import shutil


def convert_video_to_fps(fps):
    files = list(DATA_DIR.glob("Sessions/*/*FaceNear2*"))
    for file in tqdm(files, desc=f"Converting videos to {fps}fps",):
        session = file.parent.name
        participant = "P1" if "P1" in file.name else "P2"
        new_file = DATASET_DIR / session / participant / f"video_{fps}fps.mp4"
        if new_file.exists():
            continue

        with tempfile.TemporaryDirectory() as tmpd:
            tmpf = Path(tmpd) / "video.mp4"
            ffmpeg.input(str(file)).output(str(tmpf), r=str(fps), vsync="1").run(
                quiet=True
            )
            new_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(tmpf, new_file)


def extract_imgs_from_videos(fps):
    msg = "Extracing images from videos"
    for file in tqdm(list(DATASET_DIR.glob(f"*/*/video_{fps}fps.mp4")), desc=msg):
        img_dir = file.parent / f"imgs_{fps}fps"

        if img_dir.exists():
            continue

        tmpd = tempfile.mkdtemp()
        ffmpeg.input(file).output(
            str(Path(tmpd) / "%5d.jpg"), format="image2", vcodec="mjpeg", qscale=0
        ).run(quiet=True)
        shutil.move(tmpd, img_dir)

