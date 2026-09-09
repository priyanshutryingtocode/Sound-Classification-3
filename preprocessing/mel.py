from pathlib import Path

import pandas as pd
from PIL import Image

from .config import CONFIG
from .representations import (
    load_audio,
    compute_mel,
    min_max_normalize
)


PROJECT_ROOT = Path("D:/BTP/Sound Classification 3")
DATASET_ROOT = Path("D:/BTP/Dataset/UrbanSound8K")

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dsvnet"
    / "separate"
    / "mel"
)


def resize_feature(feature):
    """
    Normalize feature to [0, 1], resize to the configured
    image size, and convert to uint8 grayscale.
    """

    feature = min_max_normalize(feature)

    image = Image.fromarray(
        feature.astype("float32"),
        mode="F"
    )

    image = image.resize(
        CONFIG.image_size,
        resample=Image.Resampling.BILINEAR
    )

    image = (
        __import__("numpy")
        .asarray(image, dtype="float32")
    )

    image = (
        image * 255.0
    ).clip(0, 255)

    return __import__("numpy").rint(image).astype("uint8")


def process_dataset():

    metadata_path = (
        DATASET_ROOT
        / "metadata"
        / "UrbanSound8K.csv"
    )

    metadata = pd.read_csv(metadata_path)

    processed = 0
    failed = 0

    for _, row in metadata.iterrows():

        fold = int(row["fold"])
        class_name = row["class"]
        file_name = row["slice_file_name"]

        audio_path = (
            DATASET_ROOT
            / "audio"
            / f"fold{fold}"
            / file_name
        )

        output_dir = (
            OUTPUT_ROOT
            / f"fold{fold}"
            / class_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            output_dir
            / f"{Path(file_name).stem}.png"
        )

        try:

            y, sr = load_audio(audio_path)

            mel = compute_mel(y, sr)

            image = resize_feature(mel)

            Image.fromarray(
                image,
                mode="L"
            ).save(output_path)

            processed += 1

            if processed % 500 == 0:
                print(
                    f"Processed: {processed}/{len(metadata)}"
                )

        except Exception as e:

            failed += 1

            print(
                f"FAILED: {file_name} -> {e}"
            )

    print("\nMEL PROCESSING COMPLETE")
    print(f"Processed: {processed}")
    print(f"Failed:    {failed}")


if __name__ == "__main__":
    process_dataset()