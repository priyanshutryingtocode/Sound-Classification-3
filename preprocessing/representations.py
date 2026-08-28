import numpy as np
import librosa

from .config import CONFIG


def load_audio(audio_path):
    """
    Load an UrbanSound8K audio file.

    Returns
    -------
    y : np.ndarray
        Mono waveform.
    sr : int
        Sample rate.
    """

    y, sr = librosa.load(
        audio_path,
        sr=CONFIG.sample_rate,
        mono=CONFIG.mono
    )

    return y, sr


def min_max_normalize(x):
    """
    Normalize a spectrogram/feature to [0, 1].
    """

    x = np.asarray(x, dtype=np.float32)

    min_val = np.min(x)
    max_val = np.max(x)

    if max_val - min_val < 1e-8:
        return np.zeros_like(x, dtype=np.float32)

    return (
        (x - min_val)
        / (max_val - min_val)
    ).astype(np.float32)


def compute_mel(y, sr):
    """
    Compute MEL spectrogram.
    """

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=CONFIG.n_fft,
        hop_length=CONFIG.hop_length,
        n_mels=CONFIG.n_mels,
        power=2.0
    )

    # Convert power to dB for visualization/features.
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db


def compute_stft(y, sr):
    """
    Compute magnitude STFT in dB.
    """

    stft = librosa.stft(
        y,
        n_fft=CONFIG.n_fft,
        hop_length=CONFIG.hop_length
    )

    magnitude = np.abs(stft)

    stft_db = librosa.amplitude_to_db(
        magnitude,
        ref=np.max
    )

    return stft_db


def compute_cqt(y, sr):
    """
    Compute CQT magnitude in dB.
    """

    cqt = librosa.cqt(
        y=y,
        sr=sr,
        hop_length=CONFIG.hop_length,
        n_bins=CONFIG.cqt_n_bins,
        bins_per_octave=CONFIG.cqt_bins_per_octave
    )

    magnitude = np.abs(cqt)

    cqt_db = librosa.amplitude_to_db(
        magnitude,
        ref=np.max
    )

    return cqt_db


def compute_chromagram(y, sr):
    """
    Compute CQT-based chromagram.
    """

    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        hop_length=CONFIG.hop_length
    )

    return chroma