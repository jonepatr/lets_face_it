from glow_pytorch.generate_motion_from_model import generate_motion
from misc.shared import BASE_DIR
from misc.generate_test_sequences import generate_videos, get_vad_weights
from misc.find_test_segments import LengthFail, get_frames
import random
import torch

PADDING = 24 * 2

def render(name, output_folder, session, start, stop, info):
    start = start
    stop = stop + stop % 2

    frames = get_frames(session, start, stop)

    p1_vad = get_vad_weights("P1", session, start, stop).sum()
    p2_vad = get_vad_weights("P2", session, start, stop).sum()

    
    info = info.copy()

    if p1_vad > p2_vad:
        info["left_start"] = 0
        info["right_start"] = 136
    else:
        info["left_start"] = 136
        info["right_start"] = 0

    frame_nbs = [str(x).zfill(5) for x in sorted(frames[:, -1].int().tolist())]

    left_video = {
        "expression": frames[:, info["left_start"] : info["left_start"] + 50],
        "jaw": frames[:, info["left_start"] + 100 : info["left_start"] + 103],
        "neck": frames[:, info["left_start"] + 103 : info["left_start"] + 106],
    }

    # the agent on the right side is the agent that is p1, i.e. "self"
    p1_indicies = list(range(info["right_start"], info["right_start"] + 136))

    # the agent on the left side of the video is p2, i.e. the interloctur
    p2_indicies = list(range(info["left_start"], info["left_start"] + 136))


    try:
        # 24 is the number of frames needed for initialization of the model, and * 2 because we only use every second frame
        special_frames = get_frames(session, start - PADDING, stop)
        
    except (LengthFail, IndexError) as e:
        print(f"failed fetching frames.. {e}")
        return
        
        

    p1_model_frames = special_frames[:, p1_indicies]
    p2_model_frames = special_frames[:, p2_indicies]
    

    predicted_sequence = generate_motion(torch.cat([p1_model_frames, p2_model_frames], dim=1), model_path=BASE_DIR / "models/final_model.ckpt")
    

    right_video = {
        "expression": predicted_sequence[0, :, :50],
        "jaw": predicted_sequence[0, :, 100:103],
        "neck": predicted_sequence[0, :, 103:106],
    }

    generate_videos(
        [(name, session, left_video, right_video, info, frame_nbs)], output_folder, 2, overwrite=True
    )



if __name__ == '__main__':
    info = {
        "left_gender": random.choice(["male", "female"]),
        "right_gender": random.choice(["male", "female"]),
        "left_shape": torch.randn(300).tolist(),
        "right_shape": torch.randn(300).tolist(),
        "left_skin_color": random.choice(["white", "black"]),
        "right_skin_color": random.choice(["white", "black"]),
    }

    start_frame = 16867+2000
    length = 600
    for i in range(4):
        render(f"3_new_hello_{i}.mp4", BASE_DIR / "outputs/fresh_baked", "25", start_frame, start_frame+length, info)

    