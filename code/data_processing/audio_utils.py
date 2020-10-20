import os
from pathlib import Path

import librosa
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig
import soundfile
from scipy.signal._savitzky_golay import savgol_filter
from tqdm import tqdm

import parselmouth as pm
from misc.shared import DATA_DIR
from pydub import AudioSegment


def compute_prosody(audio_filename, time_step=0.05):
    audio = pm.Sound(audio_filename)

    # Extract pitch and intensity
    pitch = audio.to_pitch(time_step=time_step)
    intensity = audio.to_intensity(time_step=time_step)

    # Evenly spaced time steps
    times = np.arange(0, audio.get_total_duration() - time_step, time_step)

    # Compute prosodic features at each time step
    pitch_values = np.nan_to_num(
        np.asarray([pitch.get_value_at_time(t) for t in times])
    )
    intensity_values = np.nan_to_num(
        np.asarray([intensity.get_value(t) for t in times])
    )

    intensity_values = np.clip(
        intensity_values, np.finfo(intensity_values.dtype).eps, None
    )

    # Normalize features [Chiu '11]
    pitch_norm = np.clip(np.log(pitch_values + 1) - 4, 0, None)
    intensity_norm = np.clip(np.log(intensity_values) - 3, 0, None)

    return pitch_norm, intensity_norm


def derivative(x, f):
    """ Calculate numerical derivative (by FDM) of a 1d array
    Args:
        x: input space x
        f: Function of x
    Returns:
        der:  numerical derivative of f wrt x
    """

    x = 1000 * x  # from seconds to milliseconds

    # Normalization:
    dx = x[1] - x[0]

    cf = np.convolve(f, [1, -1]) / dx

    # Remove unstable values
    der = cf[:-1].copy()
    der[0] = 0

    return der


def extract_prosodic_features(audio_filename, time_step=0.02):
    """
    Extract all 5 prosodic features
    Args:
        audio_filename:   file name for the audio to be used
    Returns:
        pros_feature:     energy, energy_der, pitch, pitch_der, pitch_ind
    """

    # Read audio from file
    sound = AudioSegment.from_file(audio_filename, format="wav")

    # Alternative prosodic features
    pitch, energy = compute_prosody(audio_filename, time_step)

    duration = len(sound) / 1000
    t = np.arange(0, duration, time_step)

    energy_der = derivative(t, energy)
    pitch_der = derivative(t, pitch)

    # Stack them all together
    pros_feature = np.stack((energy, energy_der, pitch, pitch_der))

    # And reshape
    pros_feature = np.transpose(pros_feature)

    return pros_feature


def split_audio_channels():
    for file_ in DATA_DIR.glob("Sessions/*/*c1_c2.wav"):
        wav_dir = DATA_DIR / f"Sessions_wav/{file_.parent.name}"
        wav_dir.mkdir(parents=True, exist_ok=True)
        data, fs = soundfile.read(file_)
        soundfile.write(
            wav_dir / "P1.wav", data[:, 0], fs, "PCM_16",
        )
        soundfile.write(
            wav_dir / "P2.wav", data[:, 1], fs, "PCM_16",
        )


def prepare_audio_segments_v2(audio_path: Path, dest_path: Path):
    if dest_path.exists():
        return True

    y, fs = librosa.load(audio_path, 48000, mono=False)

    for channel in range(2):
        single_y = y[channel]
        segments = librosa.effects.split(single_y, top_db=3)

        prev_seg_start = 0
        participant = "P" + str(channel + 1)

        save = lambda x, y: soundfile.write(x, y, fs, "PCM_16")
        save_path = dest_path / participant
        save_path.mkdir(parents=True)
        for i, segment in enumerate(segments):
            data = single_y[prev_seg_start : segment[0]]
            save(save_path / f"{i:05}.wav", data)
            prev_seg_start = segment[0]

        save(save_path / f"{i+1:05}.wav", single_y[prev_seg_start:])


def prepare_audio_segments(audio_paths, audio_channel, dest_path):
    if os.path.exists(dest_path):
        return True

    os.makedirs(dest_path, exist_ok=True)
    for n, audio_path in enumerate(tqdm(audio_paths)):
        y, fs = librosa.load(audio_path, 48000, mono=False)

        single_y = y[audio_channel]
        segments = librosa.effects.split(single_y, top_db=3)

        prev_seg_start = 0

        for i, segment in enumerate(segments):
            soundfile.write(
                os.path.join(dest_path, f"{n:06}_{i:03}.wav"),
                single_y[prev_seg_start : segment[0]],
                fs,
                "PCM_16",
            )
            prev_seg_start = segment[0]
        soundfile.write(
            os.path.join(dest_path, str(i + 1).zfill(3) + ".wav"),
            single_y[prev_seg_start:],
            fs,
            "PCM_16",
        )


def crosstalk_vad(
    speaker1_path, speaker2_path, frame_count, tha=30, thb=5, savgol_win=301, savgol_poly_order=1
):
    """
    tha: absolute dB level for when to consider there to be speech activity in a channel
    thb: minimum difference between channels to consider it to be one speaker only
    """

    fs, x1 = wav.read(speaker1_path)
    _, x2 = wav.read(speaker2_path)

    x1 = x1.astype("float")
    x2 = x2.astype("float")

    # calculate rms energy in dB at a rate of 100 Hz (hop length 0.01 s)
    e1 = librosa.core.amplitude_to_db(
        librosa.feature.rms(x1, frame_length=int(fs * 0.02), hop_length=int(fs * 0.01))
    ).flatten()
    e2 = librosa.core.amplitude_to_db(
        librosa.feature.rms(x2, frame_length=int(fs * 0.02), hop_length=int(fs * 0.01))
    ).flatten()

    # boolean vectors at 100 Hz, s1: only speaker 1. s2: only speaker 2.
    s1 = np.logical_and(np.greater(e1, tha), np.greater(e1, e2 + thb))
    s2 = np.logical_and(np.greater(e2, tha), np.greater(e2, e1 + thb))

    smooth_s1 = savgol_filter(s1, savgol_win, savgol_poly_order,)
    smooth_s2 = savgol_filter(s2, savgol_win, savgol_poly_order,)

    s1x = np.clip(sig.resample(smooth_s1, frame_count, window="hamming"), 0, 1)
    s2x = np.clip(sig.resample(smooth_s2, frame_count, window="hamming"), 0, 1)

    s1x[s1x >= 0.1] = 1
    s2x[s2x >= 0.1] = 1

    s1x[s1x < 0.1] = 0
    s2x[s2x < 0.1] = 0
    
    return s1x, s2x
