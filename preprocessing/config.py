from dataclasses import dataclass


@dataclass
class AudioConfig:

    sample_rate: int = 44100
    mono: bool = True

    n_fft: int = 2048
    hop_length: int = 512

    n_mels: int = 128

    cqt_n_bins: int = 84
    cqt_bins_per_octave: int = 12

    image_size: tuple[int, int] = (224, 224)


CONFIG = AudioConfig()