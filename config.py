from pathlib import Path


# ============================================================
# Dataset
# ============================================================

DATASET_ROOT = Path(
    r"D:\BTP\Dataset\UrbanSound8K"
)


# ============================================================
# Project data
# ============================================================

PROCESSED_ROOT = Path("data") / "processed"

DSVNET_ROOT = (
    PROCESSED_ROOT / "dsvnet"
)

DSVNET_RGB_ROOT = (
    DSVNET_ROOT / "rgb"
)

DSVNET_MANIFEST = (
    DSVNET_ROOT / "manifest.csv"
)


# ============================================================
# Training
# ============================================================

NUM_CLASSES = 10

IMAGE_SIZE = 224