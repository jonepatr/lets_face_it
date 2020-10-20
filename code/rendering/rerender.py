from misc.shared import BASE_DIR
from misc.generate_test_sequences import generate_videos, get_vad_weights, get_vocas
import json
from misc.find_test_segments import get_frames, LengthFail
from tqdm import tqdm
from glow_pytorch.generate_motion_from_model import generate_motion
import torch



PADDING = 24 * 2


# mimicry_no_mimicry_rand_align
def rerendering(in_folder, out_folder, txt_folder, prefix, model=False):
    fail_count = 0
    vids = []
    vv = list((BASE_DIR / "outputs/vids").glob(f"{in_folder}/*.mp4"))
    for file_ in tqdm(vv):

        try:
            name, session, start, stop, start2, stop2 = file_.stem.rsplit("_", 5)


            orig_start2 = start2
            orig_stop2 = stop2
            start2 = int(start2)
            stop2 = int(stop2) + int(stop2) % 2

            frames2 = get_frames(session, start2, stop2)
        except ValueError:
            name, session, start, stop = file_.stem.rsplit("_", 3)
            frames2 = None


        orig_start = start
        orig_stop = stop
        start = int(start)
        stop = int(stop) + int(stop) % 2

        code = f"{session}_{start}_{stop-1}"
        txt_file = list(file_.parent.parent.glob(f"{txt_folder}/meta/*{code}*"))
        # if not txt_file:
        #     continue
        f = txt_file[0]

        info = json.loads((f).with_suffix(".txt").read_text())
        frames = get_frames(session, start, stop)

        p1_vad = get_vad_weights("P1", session, start, stop).sum()
        p2_vad = get_vad_weights("P2", session, start, stop).sum()

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


        if model:

            # the agent on the right side is the agent that is p1, i.e. "self"
            p1_indicies = list(range(info["right_start"], info["right_start"] + 136))

            # the agent on the left side of the video is p2, i.e. the interloctur
            p2_indicies = list(range(info["left_start"], info["left_start"] + 136))
            
            try:
                # 24 is the number of frames needed for initialization of the model, and * 2 because we only use every second frame
                special_frames = get_frames(session, (start - PADDING), stop)
                if frames2 is not None:
                    special_frames2 = get_frames(session, (start2 - PADDING), stop2)
                else:
                    special_frames2 = special_frames
            except (LengthFail, IndexError) as e:
                print(f"failed fetching frames.. {file_}; {e}")
                fail_count += 1
                continue

            p1_model_frames = special_frames[:, p1_indicies]

            p2_model_frames = special_frames2[:, p2_indicies]

            predicted_sequence = model(
                torch.cat([p1_model_frames, p2_model_frames], dim=1)
            )

            right_video = {
                "expression": predicted_sequence[0, :, :50],
                "jaw": predicted_sequence[0, :, 100:103],
                "neck": predicted_sequence[0, :, 103:106],
            }
        else:
            if frames2 is not None:
                p2_frames = frames2
            else:
                p2_frames = frames

            right_video = {
                "expression": p2_frames[
                    :, info["right_start"] : info["right_start"] + 50
                ],
                "jaw": p2_frames[
                    :, info["right_start"] + 100 : info["right_start"] + 103
                ],
                "neck": p2_frames[
                    :, info["right_start"] + 103 : info["right_start"] + 106
                ],
            }

        vids.append(
            (f"{prefix}_{file_.name}", session, left_video, right_video, info, frame_nbs)
        )

    generate_videos(
        vids, BASE_DIR / "outputs/outs" / out_folder, 1.7, overwrite=False
    )
    print("failed:", fail_count)


def run_model(frames):
    model_path = BASE_DIR / "models/final_model.ckpt"
    return generate_motion(frames, model_path=model_path, eps=0.3)
