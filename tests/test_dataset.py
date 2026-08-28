from pathlib import Path

from datasets.urbansound import UrbanSound8KDataset


DATASET_ROOT = r"D:\BTP\Dataset\UrbanSound8K"


def main():

    print("=" * 60)
    print("UrbanSound8K Dataset Test")
    print("=" * 60)

    # --------------------------------------------------
    # Test complete dataset
    # --------------------------------------------------

    dataset = UrbanSound8KDataset(
        DATASET_ROOT
    )

    print(f"\nMetadata file:")
    print(dataset.metadata_path)

    print(f"\nTotal samples: {len(dataset)}")

    assert len(dataset) == 8732, (
        f"Expected 8732 samples, "
        f"got {len(dataset)}"
    )

    print("✓ Complete dataset loaded")

    # --------------------------------------------------
    # Test individual sample
    # --------------------------------------------------

    waveform, label, metadata = dataset[0]

    print("\nFirst sample:")
    print(f"Waveform shape: {tuple(waveform.shape)}")
    print(f"Label:          {label}")
    print(f"Class:          {metadata['class_name']}")
    print(f"Fold:           {metadata['fold']}")
    print(f"Sample rate:    {metadata['sample_rate']}")
    print(f"File:           {metadata['filename']}")

    assert waveform.ndim == 2
    assert 0 <= label <= 9
    assert 1 <= metadata["fold"] <= 10

    print("✓ Individual sample loaded")

    # --------------------------------------------------
    # Test fold filtering
    # --------------------------------------------------

    fold1 = UrbanSound8KDataset(
        DATASET_ROOT,
        folds=[1]
    )

    print(
        f"\nFold 1 samples: {len(fold1)}"
    )

    assert all(
        fold == 1
        for fold in fold1.metadata["fold"]
    )

    print("✓ Fold filtering works")

    # --------------------------------------------------
    # Test multiple folds
    # --------------------------------------------------

    folds_1_2 = UrbanSound8KDataset(
        DATASET_ROOT,
        folds=[1, 2]
    )

    print(
        f"Folds 1 + 2 samples: "
        f"{len(folds_1_2)}"
    )

    assert set(
        folds_1_2.metadata["fold"]
    ) == {1, 2}

    print("✓ Multiple-fold filtering works")

    print("\n" + "=" * 60)
    print("ALL DATASET TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()