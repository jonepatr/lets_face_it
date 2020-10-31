import ffmpeg

def count_video_frames(file):
    return int(ffmpeg.probe(file)["streams"][0]["nb_frames"])