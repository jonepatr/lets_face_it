import shutil
import tempfile
from pathlib import Path

import librosa
import numpy as np
import parselmouth as pm
import scipy.io.wavfile as wav
import scipy.signal as sig
import soundfile
from misc.shared import DATA_DIR, DATASET_DIR
from pydub import AudioSegment
from python_speech_features import mfcc
from scipy.signal._savitzky_golay import savgol_filter
from tqdm import tqdm

from feature_extraction.shared import count_video_frames


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


def extract_prosodic_features(audio_filename, nb_frames, time_step=0.02):
    """
    Extract all 4 prosodic features
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

    return sig.resample(pros_feature, nb_frames)


def split_audio_channels():
    files = list(DATA_DIR.glob("Sessions/*/*c1_c2.wav"))
    for file in tqdm(files, desc="Splitting audio files"):
        session = file.parent.name
        data = None

        for i, participant in enumerate(("P1", "P2")):
            wav_file = DATASET_DIR / session / participant / "audio.wav"
            if wav_file.exists():
                continue
            if data is None:
                data, fs = soundfile.read(file)

            with tempfile.TemporaryDirectory() as tmpd:
                tmpf = Path(tmpd) / "audio.wav"
                soundfile.write(tmpf, data[:, i], fs, "PCM_16")
                wav_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(tmpf, wav_file)


def chunk_audio():
    files = list(DATASET_DIR.glob("*/*/audio.wav"))

    for file in tqdm(files, desc="Chunking audio"):
        audio_chunk_dir = file.parent / "audio_chunks"
        if audio_chunk_dir.exists():
            continue

        y, fs = librosa.load(file, librosa.get_samplerate(file))
        segments = librosa.effects.split(y, top_db=3)

        prev_seg_start = 0
        tmpd = Path(tempfile.mkdtemp())
        for i, segment in enumerate(segments, 1):
            data = y[prev_seg_start : segment[0]]
            soundfile.write(tmpd / f"{i:05}.wav", data, fs, "PCM_16")
            prev_seg_start = segment[0]
        soundfile.write(tmpd / f"{i+1:05}.wav", y[prev_seg_start:], fs, "PCM_16")

        shutil.move(tmpd, audio_chunk_dir)


def crosstalk_vad(
    speaker1_path,
    speaker2_path,
    frame_count,
    tha=30,
    thb=5,
    savgol_win=301,
    savgol_poly_order=1,
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


def extract_prosody(fps):
    files = list(DATASET_DIR.glob("*/*/audio.wav"))

    for file in tqdm(files, desc="Extracting prosodic features"):
        prosody_file = file.parent / f"prosodic_features_{fps}fps.npy"
        video_file = file.parent / f"video_{fps}fps.mp4"

        if prosody_file.exists() or not video_file.exists():
            continue

        nb_frames = count_video_frames(video_file)

        # Calculate prosodic features
        prosodic_featuers = extract_prosodic_features(str(file), nb_frames)

        np.save(prosody_file, prosodic_featuers)


def extract_mfcc(fps, num_cep=26, window_length=0.02, window_step=0.01, nfft=1024):
    files = list(DATASET_DIR.glob("*/*/audio.wav"))

    for file in tqdm(files, desc="Extracting mfccs"):
        mfcc_file = file.parent / f"mfcc_{fps}fps.npy"
        video_file = file.parent / f"video_{fps}fps.mp4"

        if mfcc_file.exists() or not video_file.exists():
            continue

        nb_frames = count_video_frames(video_file)

        # Calculate MFCC feature with the window frame it was designed for
        fs, audio = wav.read(file)
        mfcc_features = mfcc(
            audio,
            winlen=window_length,
            winstep=window_step,
            samplerate=fs,
            numcep=num_cep,
            nfft=nfft,
        )

        # Resample to desired frame length
        resampled_mfccs = sig.resample(mfcc_features, nb_frames)

        np.save(mfcc_file, resampled_mfccs)


def extract_vad(fps):
    sessions = list(DATASET_DIR.glob("*"))

    for session in tqdm(sessions, desc="Extracting crosstalk vad"):
        p1_wav = session / "P1" / "audio.wav"
        p2_wav = session / "P2" / "audio.wav"

        p1_vad_file = session / "P1" / f"crosstalk_vad_{fps}fps.npy"
        p2_vad_file = session / "P2" / f"crosstalk_vad_{fps}fps.npy"

        p1_video_file = session / "P1" / f"video_{fps}fps.mp4"
        p2_video_file = session / "P2" / f"video_{fps}fps.mp4"

        if p1_vad_file.exists() or p2_vad_file.exists():
            continue

        if not p1_video_file.exists() or not p2_video_file.exists():
            continue

        p1_nb_frames = count_video_frames(p1_video_file)
        p2_nb_frames = count_video_frames(p2_video_file)

        assert p1_nb_frames == p2_nb_frames

        p1, p2 = crosstalk_vad(p1_wav, p2_wav, p1_nb_frames)

        np.save(p1_vad_file, p1)
        try:
            np.save(p2_vad_file, p2)
        except:
            p1_vad_file.unlink()

