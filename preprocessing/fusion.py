import numpy as np
from PIL import Image

from .config import CONFIG
from .representations import (
    min_max_normalize
)


def resize_feature(feature):
    """
    Resize a 2D feature map to 224 x 224.

    The feature is normalized before resizing.
    """

    feature = min_max_normalize(feature)

    # Convert [0, 1] -> [0, 255]
    feature_uint8 = (
        feature * 255
    ).clip(0, 255).astype(np.uint8)

    image = Image.fromarray(
        feature_uint8,
        mode="L"
    )

    image = image.resize(
        CONFIG.image_size,
        resample=Image.Resampling.BILINEAR
    )

    return np.asarray(
        image,
        dtype=np.uint8
    )


def create_rgb_fusion(
    mel,
    stft,
    cqt
):
    """
    Create the paper's RGB-like fusion:

        R = MEL
        G = STFT
        B = CQT

    Each representation is independently
    normalized to [0, 1] before conversion
    to 8-bit [0, 255].
    """

    mel_norm = min_max_normalize(mel)
    stft_norm = min_max_normalize(stft)
    cqt_norm = min_max_normalize(cqt)

    # Resize each independently.
    mel_img = Image.fromarray(
        (mel_norm * 255).clip(0, 255).astype(np.uint8),
        mode="L"
    ).resize(
        CONFIG.image_size,
        resample=Image.Resampling.BILINEAR
    )

    stft_img = Image.fromarray(
        (stft_norm * 255).clip(0, 255).astype(np.uint8),
        mode="L"
    ).resize(
        CONFIG.image_size,
        resample=Image.Resampling.BILINEAR
    )

    cqt_img = Image.fromarray(
        (cqt_norm * 255).clip(0, 255).astype(np.uint8),
        mode="L"
    ).resize(
        CONFIG.image_size,
        resample=Image.Resampling.BILINEAR
    )

    # Stack:
    #
    # R = MEL
    # G = STFT
    # B = CQT

    rgb = np.stack(
        [
            np.asarray(mel_img),
            np.asarray(stft_img),
            np.asarray(cqt_img)
        ],
        axis=-1
    )

    return rgb.astype(np.uint8)