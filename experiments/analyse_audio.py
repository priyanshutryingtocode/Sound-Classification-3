from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path(
    r"D:\BTP\Dataset\UrbanSound8K"
)

METADATA_PATH = (
    DATASET_ROOT
    / "metadata"
    / "UrbanSound8K.csv"
)

OUTPUT_PATH = (
    Path("results")
    / "audio_audit.csv"
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("UrbanSound8K Audio Audit")
    print("=" * 70)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    metadata = pd.read_csv(METADATA_PATH)

    print(f"\nMetadata samples: {len(metadata)}")

    # --------------------------------------------------------
    # Analyze every audio file
    # --------------------------------------------------------

    records = []

    for _, row in tqdm(
        metadata.iterrows(),
        total=len(metadata),
        desc="Analyzing audio"
    ):

        fold = int(row["fold"])
        filename = row["slice_file_name"]

        audio_path = (
            DATASET_ROOT
            / "audio"
            / f"fold{fold}"
            / filename
        )

        try:

            info = sf.info(str(audio_path))

            duration = (
                info.frames / info.samplerate
            )

            records.append({
                "filename": filename,
                "fold": fold,
                "classID": int(row["classID"]),
                "class": row["class"],
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "duration": duration,
            })

        except Exception as e:

            print(
                f"\nERROR: {audio_path}"
            )
            print(e)

    df = pd.DataFrame(records)

    # --------------------------------------------------------
    # Save raw audit results
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Duration statistics
    # --------------------------------------------------------

    duration = df["duration"]

    print("\n" + "=" * 70)
    print("DURATION STATISTICS")
    print("=" * 70)

    print(
        f"Minimum:          {duration.min():.4f} sec"
    )

    print(
        f"25th percentile:  {duration.quantile(0.25):.4f} sec"
    )

    print(
        f"Median:           {duration.median():.4f} sec"
    )

    print(
        f"Mean:             {duration.mean():.4f} sec"
    )

    print(
        f"75th percentile:  {duration.quantile(0.75):.4f} sec"
    )

    print(
        f"Maximum:          {duration.max():.4f} sec"
    )

    # --------------------------------------------------------
    # Short files
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SHORT AUDIO FILES")
    print("=" * 70)

    thresholds = [
        0.10,
        0.25,
        0.50,
        1.00,
        2.00,
    ]

    for threshold in thresholds:

        count = (
            duration < threshold
        ).sum()

        percentage = (
            count / len(df) * 100
        )

        print(
            f"< {threshold:.2f} sec: "
            f"{count:5d} files "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Sample rates
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAMPLE RATES")
    print("=" * 70)

    sample_rates = (
        df["sample_rate"]
        .value_counts()
        .sort_index()
    )

    for rate, count in sample_rates.items():

        percentage = (
            count / len(df) * 100
        )

        print(
            f"{rate} Hz: "
            f"{count:5d} files "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Channel counts
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CHANNEL COUNTS")
    print("=" * 70)

    channel_counts = (
        df["channels"]
        .value_counts()
        .sort_index()
    )

    for channels, count in channel_counts.items():

        percentage = (
            count / len(df) * 100
        )

        print(
            f"{channels} channel(s): "
            f"{count:5d} files "
            f"({percentage:.2f}%)"
        )

    # --------------------------------------------------------
    # Duration by class
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DURATION BY CLASS")
    print("=" * 70)

    class_stats = (
        df.groupby("class")["duration"]
        .agg(
            count="count",
            min="min",
            mean="mean",
            median="median",
            max="max",
        )
        .sort_values("mean")
    )

    print(
        class_stats.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # --------------------------------------------------------
    # Duration by fold
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DURATION BY FOLD")
    print("=" * 70)

    fold_stats = (
        df.groupby("fold")["duration"]
        .agg(
            count="count",
            min="min",
            mean="mean",
            median="median",
            max="max",
        )
    )

    print(
        fold_stats.to_string(
            float_format=lambda x: f"{x:.3f}"
        )
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(
        f"Successfully analyzed: {len(df)} / {len(metadata)}"
    )

    missing = len(metadata) - len(df)

    print(
        f"Failed/missing:        {missing}"
    )

    if len(df) != len(metadata):

        print(
            "\nWARNING: Some audio files could "
            "not be analyzed."
        )

    else:

        print(
            "✓ All 8,732 audio files analyzed"
        )

    print(
        f"\nDetailed results saved to:"
    )

    print(
        OUTPUT_PATH.resolve()
    )

    print("\n" + "=" * 70)
    print("AUDIO AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()