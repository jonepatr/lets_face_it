from misc.shared import BASE_DIR, DATA_DIR
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from misc.utils import get_gender, get_participant, ms2frames, replace_part
from feature_extraction.mesh_utils import get_flame_parameters_for_objs
import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm


class Segment:
    @property
    def duration_ms(self):
        return self.stop_ms - self.start_ms

    def __getattr__(self, name):
        if name.endswith("_s"):
            result = getattr(self, name[:-2] + "_ms")
            if result is not None:
                return result / 1000.0
        elif name.endswith("_frames"):
            result = getattr(self, name[:-7] + "_ms")
            return ms2frames(result, fps=50)

    # def get_files(self, files, only_odd=False, padding_ms=0):

    def get_voca_flame_param_files(self, frames, participant):

        dest_dir = frames[0].parents[1] / "flame_params"
        gender = get_gender(self.session, participant)
        model_fname = BASE_DIR / f"models/flame_model/ch_models/{gender}_model.pkl"

        get_flame_parameters_for_objs(frames, dest_dir, model_fname)

        return [dest_dir / x.name for x in frames]

    def _get_start_stop(self, start_frames, stop_frames):
        if not start_frames:
            start_frames = self.start_frames

        if not stop_frames:
            stop_frames = self.stop_frames

        if isinstance(self, DataSegment):
            start_frames = max(self.start_frames, start_frames)
            stop_frames = min(self.stop_frames, stop_frames)

        elif isinstance(self, MimicrySegment):
            start_frames = max(self.data_segment.start_frames, start_frames)
            stop_frames = min(self.data_segment.stop_frames, stop_frames)
        else:
            raise NotImplementedError()

        return start_frames, stop_frames

    def get_flame_params(
        self,
        participant,
        type_="default",
        only_odd=False,
        start_frames=None,
        stop_frames=None,
    ):
        if type_ == "voca":
            files = DATA_DIR.glob(
                f"Sessions_50fps_voca/{self.session}/*{participant}*/vertices/*.npy"
            )
        else:
            files = DATA_DIR.glob(
                f"Sessions_50fps_flame_fitting/{self.session}/*{participant}*/*/flame_params.npy"
            )

        start_frames, stop_frames = self._get_start_stop(start_frames, stop_frames)

        frames = []
        for file_ in sorted(files):
            try:
                frame = int(file_.parent.stem)
            except ValueError:
                frame = int(file_.stem)

            if (
                (not only_odd or frame % 2 == 1)
                and frame >= start_frames
                and frame <= stop_frames
            ):
                frames.append(file_)

        if type_ == "voca":
            frames = self.get_voca_flame_param_files(frames, participant)

        if frames:
            return get_flame_params_for_files(frames)
        else:
            return None

    @staticmethod
    def merge_flame_params_and_voca(
        flame_params, voca_flame_params, vad_weights, window=11, polyorder=3
    ):
        smooth_pose = savgol_filter(
            flame_params["pose"], window, polyorder, axis=0
        )  # flame_params["pose"] # savgol_filter(flame_params["pose"], window, polyorder, axis=0)
        smooth_expression = savgol_filter(
            flame_params["expression"], window, polyorder, axis=0
        )  # savgol_filter(
        # flame_params["expression"], window, polyorder, axis=0
        # )

        avg_rot = flame_params["rot"].mean(axis=0)
        avg_rot[1:] = 0
        smooth_neck = (
            savgol_filter(flame_params["neck"], window, polyorder, axis=0) - avg_rot
        )
        smooth_eye = flame_params[
            "eye"
        ]  # savgol_filter(flame_params["eye"], window, polyorder, axis=0)

        shape = np.zeros((1, 300))
        shape[:, :100] = np.random.randn(100) * 1.0
        shape_params = np.repeat(shape, smooth_pose.shape[0], axis=0)

        voca_pose = voca_flame_params["pose"] * np.repeat(
            vad_weights, voca_flame_params["pose"].shape[1], axis=1
        )
        voca_expression = voca_flame_params["expression"] * np.repeat(
            vad_weights, voca_flame_params["expression"].shape[1], axis=1
        )

        return {
            "shape_params": shape_params,
            "pose_params": smooth_pose + voca_pose,
            "expression_params": smooth_expression + voca_expression,
            "neck_params": smooth_neck,
            "eye_params": smooth_eye,
        }

    def get_vad_weights(
        self, participant, only_odd=False, start_frames=None, stop_frames=None
    ):
        start_frames, stop_frames = self._get_start_stop(start_frames, stop_frames)
        vad_weights = np.load(
            (DATA_DIR / "Sessions_vad" / self.session / participant).with_suffix(".npy")
        )
        return np.expand_dims(
            vad_weights[start_frames - 1 : stop_frames - 1 : 2 if only_odd else 1], 1
        )


class DataSegment(Segment):
    def __init__(self, session, data_type, start_ms, stop_ms):
        self.session = session
        self.data_type = data_type
        self.start_ms = start_ms
        self.stop_ms = stop_ms

    def __repr__(self):
        return f"DataSegment(start_ms={self.start_ms}, stop_ms={self.stop_ms}, session={self.session}, data_type={self.data_type})"


class MimicrySegment(Segment):
    def __init__(self, mimicry_type, start_ms, stop_ms, data_segment):
        self.mimicry_type = mimicry_type
        self.start_ms = start_ms
        self.stop_ms = stop_ms
        self.data_segment = data_segment
        self.session = data_segment.session

    def __repr__(self):
        return f"MimicrySegment(mimicry_type={self.mimicry_type}, start_ms={self.start_ms}, stop_ms={self.stop_ms}, data_segment={self.data_segment})"


def get_segments_v2():
    # get train/val segments
    all_sessions = json.load((BASE_DIR / "data/train_val_test.json").open())
    # print(train_val_test_data)
    all_data = []
    for data_type, data in all_sessions.items():
        if data_type == "heldout_interaction":
            continue
        for session, segments in data.items():
            for start, stop in segments:
                all_data.append((session, data_type, int(start), int(stop)))

    return all_data


def get_flame_params_for_file(flame_params_path):
    flame_params = np.load(flame_params_path, allow_pickle=True).item()

    shape = flame_params["tf_shape"][0]
    expression = flame_params["tf_exp"][0]
    pose = np.concatenate([[0, 0, 0], flame_params["tf_pose"][0, 3:6]])
    neck = flame_params["tf_pose"][0, :3] + flame_params["tf_rot"][0]
    eye = flame_params["tf_pose"][0, 6:]
    rot = flame_params["tf_rot"][0]
    return shape, expression, pose, neck, eye, rot


def get_flame_params_for_files(frames):
    all_flame_params = defaultdict(list)

    for flame_params_path in frames:
        shape, expression, pose, neck, eye, rot = get_flame_params_for_file(
            flame_params_path
        )
        all_flame_params["shape"].append(shape)
        all_flame_params["expression"].append(expression)
        all_flame_params["pose"].append(pose)
        all_flame_params["neck"].append(neck)
        all_flame_params["eye"].append(eye)
        all_flame_params["rot"].append(rot)

    return {x: np.vstack(y) for x, y in all_flame_params.items()}


def get_segments(type_="train"):
    # get train/val segments
    all_sessions = json.load((BASE_DIR / "data/train_val_test.json").open())
    # print(train_val_test_data)

    # get sessions/mimicry types/start-stop times for valid sessions (train & val)
    all_annotations = json.load((BASE_DIR / "data/annotations.json").open())

    valid_annotations = []
    session_videos = Counter()
    for session, annotations in all_annotations.items():
        valid_times = sorted(all_sessions[type_].get(session, []))

        last_start = 0
        for valid_start, valid_stop in sorted(valid_times):
            data_segment = DataSegment(session, type_, valid_start, valid_stop)
            for mimicry_type, timestamps in annotations.items():
                for start, stop, value in sorted(timestamps):
                    if start >= valid_start and stop <= valid_stop:
                        valid_annotations.append(
                            MimicrySegment(None, last_start, start - 1, data_segment)
                        )
                        last_start = stop + 1
                        valid_annotations.append(
                            MimicrySegment(mimicry_type, start, stop, data_segment)
                        )

                        session_videos[session] += 1

            valid_annotations.append(
                MimicrySegment(None, last_start, valid_stop, data_segment)
            )

    return valid_annotations
    # return [x for x in valid_annotations if session_videos[x.session] > 1]
