from feature_extraction.audio_utils import (
    extract_mfcc,
    extract_prosody,
    extract_vad,
    split_audio_channels,
    chunk_audio
)
from feature_extraction.openface import extract_openface
from feature_extraction.ringnet import extract_neutral_mesh, extract_ringnet
from feature_extraction.video_utils import (
    convert_video_to_fps,
    extract_imgs_from_videos,
)
from feature_extraction.voca import extract_voca

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