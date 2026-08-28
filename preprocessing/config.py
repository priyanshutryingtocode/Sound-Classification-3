from dataclasses import dataclass


@dataclass
class AudioConfig:
    # --------------------------------------------------
    # Audio
    # --------------------------------------------------

    # None = preserve original sample rate
    sample_rate: int | None = None

    # Convert stereo -> mono
    mono: bool = True

    # --------------------------------------------------
    # Common time-frequency parameters
    # --------------------------------------------------

    n_fft: int = 2048
    hop_length: int = 512

    # --------------------------------------------------
    # MEL
    # --------------------------------------------------

    n_mels: int = 128

    # --------------------------------------------------
    # CQT
    # --------------------------------------------------

    cqt_n_bins: int = 84
    cqt_bins_per_octave: int = 12

    # --------------------------------------------------
    # Output image
    # --------------------------------------------------

    image_size: tuple[int, int] = (224, 224)


CONFIG = AudioConfig()