import numpy as np
from PIL import Image

from .config import CONFIG
from .representations import min_max_normalize


def resize_feature(feature):
    """
    Normalize a 2D feature to [0, 1] and resize
    it to the configured image size.

    Returns:
        np.ndarray: uint8 grayscale image
    """

    feature = min_max_normalize(feature)

    image = Image.fromarray(
        feature.astype(np.float32),
        mode="F"
    )

    image = image.resize(
        CONFIG.image_size,
        resample=Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = (
        image * 255.0
    ).clip(0, 255)

    return np.rint(image).astype(np.uint8)


def create_rgb_fusion(mel, stft, cqt):
    """
    Create the DSVNet RGB fusion.

    Paper mapping:

        R = MEL
        G = STFT
        B = CQT

    Each representation is independently
    min-max normalized to [0, 1], resized,
    and converted to 8-bit [0, 255].
    """

    mel_img = resize_feature(mel)
    stft_img = resize_feature(stft)
    cqt_img = resize_feature(cqt)

    rgb = np.stack(
        [
            mel_img,
            stft_img,
            cqt_img
        ],
        axis=-1
    )

    return rgb.astype(np.uint8)