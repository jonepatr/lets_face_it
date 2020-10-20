import os
import ffmpeg
import logging


logger = logging.getLogger(__name__)


def vid_to_img(video_path, dest_path):
    logger.info("starting")
    if os.path.exists(dest_path):
        logger.info("path already exists")
        return True

    os.makedirs(dest_path, exist_ok=True)
    ffmpeg.input(video_path).output(
        os.path.join(dest_path, "%5d.jpg"), format="image2", vcodec="mjpeg", qscale=0
    ).run(quiet=True)
    logger.info("done")
