from code.feature_extraction.combine_features import combine_features
import ray

ray.init(num_gpus=2, num_cpus=16, log_to_driver=False)

import argparse

from feature_extraction.audio_utils import (
    chunk_audio,
    extract_mfcc,
    extract_prosody,
    extract_vad,
    split_audio_channels,
)
from feature_extraction.flame import extract_flame
from feature_extraction.openface import extract_openface
from feature_extraction.ringnet import extract_neutral_mesh, extract_ringnet
from feature_extraction.video_utils import (
    convert_video_to_fps,
    extract_imgs_from_videos,
)
from feature_extraction.voca import extract_voca

parser = argparse.ArgumentParser()
args = parser.parse_args()

fps = 25

convert_video_to_fps(fps)
extract_imgs_from_videos(fps)

# separate audio
split_audio_channels()
chunk_audio()

extract_prosody(fps)
extract_mfcc(fps)
extract_vad(fps)

extract_openface(fps)
extract_ringnet(fps)

extract_neutral_mesh(fps)
extract_voca(fps)

print("this will take a long time.......")
extract_flame(fps)

combine_features(fps)
