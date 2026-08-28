from pathlib import Path

import pandas as pd

from preprocessing.representations import (
    load_audio,
    compute_mel,
    compute_stft,
    compute_cqt,
    compute_chromagram,
)


DATASET_ROOT = Path(
    r"D:\BTP\Dataset\UrbanSound8K"
)

METADATA_PATH = (
    DATASET_ROOT
    / "metadata"
    / "UrbanSound8K.csv"
)


def main():

    metadata = pd.read_csv(METADATA_PATH)

    # --------------------------------------------------
    # Select shortest clips
    # --------------------------------------------------

    samples = []

    for _, row in metadata.iterrows():

        path = (
            DATASET_ROOT
            / "audio"
            / f"fold{int(row['fold'])}"
            / row["slice_file_name"]
        )

        samples.append(
            (
                path,
                row["slice_file_name"],
                row["class"]
            )
        )

    # Get audio durations using the audit data
    audit = pd.read_csv(
        "results/audio_audit.csv"
    )

    audit = audit.sort_values(
        "duration"
    )

    # Test representative durations
    selected = []

    for threshold in [
        0.10,
        0.25,
        0.50,
        1.00,
        4.00
    ]:

        candidates = audit[
            audit["duration"] >= threshold
        ]

        if len(candidates) > 0:

            selected.append(
                candidates.iloc[0]
            )

    print("=" * 70)
    print("SHORT-CLIP PREPROCESSING TEST")
    print("=" * 70)

    for row in selected:

        path = (
            DATASET_ROOT
            / "audio"
            / f"fold{int(row['fold'])}"
            / row["filename"]
        )

        print("\n" + "-" * 70)

        print(
            f"File:     {row['filename']}"
        )

        print(
            f"Class:    {row['class']}"
        )

        print(
            f"Duration: {row['duration']:.4f} sec"
        )

        try:

            y, sr = load_audio(
                str(path)
            )

            print(
                f"Processed SR: {sr}"
            )

            print(
                f"Samples:      {len(y)}"
            )

            mel = compute_mel(
                y, sr
            )

            stft = compute_stft(
                y, sr
            )

            cqt = compute_cqt(
                y, sr
            )

            chroma = compute_chromagram(
                y, sr
            )

            print(
                f"MEL:    {mel.shape}"
            )

            print(
                f"STFT:   {stft.shape}"
            )

            print(
                f"CQT:    {cqt.shape}"
            )

            print(
                f"Chroma: {chroma.shape}"
            )

            print("✓ All representations successful")

        except Exception as e:

            print("✗ FAILED")
            print(e)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()