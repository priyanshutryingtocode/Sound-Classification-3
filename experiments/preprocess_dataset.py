from pathlib import Path
import csv

import pandas as pd
from PIL import Image
from tqdm import tqdm

from config import (
    DATASET_ROOT,
    DSVNET_ROOT,
    DSVNET_RGB_ROOT,
    DSVNET_MANIFEST,
)

from preprocessing.representations import (
    load_audio,
    compute_mel,
    compute_stft,
    compute_cqt,
)

from preprocessing.fusion import (
    create_rgb_fusion,
)


# ============================================================
# Configuration
# ============================================================

METADATA_PATH = (
    DATASET_ROOT
    / "metadata"
    / "UrbanSound8K.csv"
)

IMAGE_SIZE = (224, 224)


# ============================================================
# Helpers
# ============================================================

def get_output_path(row):
    """
    Determine where the RGB image should be stored.
    """

    fold = int(row["fold"])
    class_name = row["class"]
    filename = row["slice_file_name"]

    return (
        DSVNET_RGB_ROOT
        / f"fold{fold}"
        / class_name
        / f"{Path(filename).stem}.png"
    )


def process_sample(row):
    """
    Process one UrbanSound8K audio sample.

    Returns:
        output_path
        error_message
    """

    fold = int(row["fold"])
    filename = row["slice_file_name"]

    audio_path = (
        DATASET_ROOT
        / "audio"
        / f"fold{fold}"
        / filename
    )

    output_path = get_output_path(row)

    # --------------------------------------------------------
    # Restartability
    # --------------------------------------------------------

    if output_path.exists():

        return output_path, None

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not audio_path.exists():

        return (
            output_path,
            f"Audio file not found: {audio_path}"
        )

    try:

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        y, sr = load_audio(
            str(audio_path)
        )

        # ----------------------------------------------------
        # Representations
        # ----------------------------------------------------

        mel = compute_mel(
            y,
            sr
        )

        stft = compute_stft(
            y,
            sr
        )

        cqt = compute_cqt(
            y,
            sr
        )

        # ----------------------------------------------------
        # RGB fusion
        # ----------------------------------------------------

        rgb = create_rgb_fusion(
            mel,
            stft,
            cqt
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if rgb.shape != (
            IMAGE_SIZE[0],
            IMAGE_SIZE[1],
            3
        ):

            raise ValueError(
                f"Unexpected RGB shape: {rgb.shape}"
            )

        if rgb.dtype != "uint8":

            raise ValueError(
                f"Unexpected RGB dtype: {rgb.dtype}"
            )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        image = Image.fromarray(
            rgb,
            mode="RGB"
        )

        image.save(
            output_path,
            format="PNG"
        )

        return output_path, None

    except Exception as e:

        return (
            output_path,
            f"{type(e).__name__}: {e}"
        )


# ============================================================
# Manifest
# ============================================================

def build_manifest(metadata):
    """
    Build a manifest containing only successfully
    generated samples.
    """

    records = []

    for _, row in metadata.iterrows():

        output_path = get_output_path(row)

        if output_path.exists():

            records.append({
                "slice_file_name": row["slice_file_name"],
                "fold": int(row["fold"]),
                "classID": int(row["classID"]),
                "class": row["class"],
                "processed_path": str(output_path),
            })

    manifest = pd.DataFrame(
        records
    )

    DSVNET_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest.to_csv(
        DSVNET_MANIFEST,
        index=False
    )

    return manifest


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("DSVNet Full Dataset Preprocessing")
    print("=" * 70)

    print(
        f"\nDataset root:\n{DATASET_ROOT}"
    )

    print(
        f"\nOutput root:\n{DSVNET_RGB_ROOT}"
    )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = pd.read_csv(
        METADATA_PATH
    ) 

    print(
        f"\nTotal samples: {len(metadata)}"
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    DSVNET_RGB_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process samples
    # --------------------------------------------------------

    successful = 0
    skipped = 0
    failed = 0

    errors = []

    progress = tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Processing",
    )

    for _, row in progress:

        output_path = get_output_path(
            row
        )

        # Existing output
        if output_path.exists():

            skipped += 1
            successful += 1

            progress.set_postfix(
                done=successful,
                failed=failed,
                skipped=skipped,
            )

            continue

        output_path, error = process_sample(
            row
        )

        if error is None:

            successful += 1

        else:

            failed += 1

            errors.append({
                "filename": row["slice_file_name"],
                "fold": int(row["fold"]),
                "class": row["class"],
                "error": error,
            })

        progress.set_postfix(
            done=successful,
            failed=failed,
            skipped=skipped,
        )

    # --------------------------------------------------------
    # Save errors
    # --------------------------------------------------------

    if errors:

        error_path = (
            DSVNET_ROOT
            / "preprocessing_errors.csv"
        )

        error_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        pd.DataFrame(
            errors
        ).to_csv(
            error_path,
            index=False
        )

        print(
            f"\nErrors saved to:\n{error_path}"
        )

    # --------------------------------------------------------
    # Build manifest
    # --------------------------------------------------------

    manifest = build_manifest(
        metadata
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)

    print(
        f"Total samples:       {len(metadata)}"
    )

    print(
        f"Successful/skipped:  {successful}"
    )

    print(
        f"Failed:              {failed}"
    )

    print(
        f"Manifest entries:    {len(manifest)}"
    )

    print(
        f"\nManifest:\n{DSVNET_MANIFEST.resolve()}"
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if (
        len(manifest) == len(metadata)
        and failed == 0
    ):

        print(
            "\n✓ All 8,732 samples processed successfully."
        )

    else:

        print(
            "\n⚠ Dataset is incomplete."
        )

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()