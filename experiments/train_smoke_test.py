from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import DSVNET_MANIFEST

from datasets.dsvnet_dataset import (
    DSVNetDataset,
)

from models.dsvnet import DSVNet

from training.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    NUM_CLASSES,
    NUM_WORKERS,
    SEED,
)

from training.trainer import (
    set_seed,
    train_one_epoch,
    evaluate,
)


def main():

    print("=" * 70)
    print("DSVNet Training Smoke Test")
    print("=" * 70)

    # ------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------

    set_seed(SEED)

    # ------------------------------------------------------
    # Device
    # ------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ------------------------------------------------------
    # Dataset
    #
    # Fold 1 = validation
    # Folds 2-10 = training
    # ------------------------------------------------------

    print("\nLoading datasets...")

    train_dataset = DSVNetDataset(
        manifest_path=DSVNET_MANIFEST,
        folds=list(range(2, 11)),
    )

    val_dataset = DSVNetDataset(
        manifest_path=DSVNET_MANIFEST,
        folds=[1],
    )

    print(
        f"Training samples:   "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    # ------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # ------------------------------------------------------
    # Verify one batch
    # ------------------------------------------------------

    print("\nChecking one batch...")

    images, labels = next(
        iter(train_loader)
    )

    print(
        f"Image batch shape: "
        f"{tuple(images.shape)}"
    )

    print(
        f"Label batch shape: "
        f"{tuple(labels.shape)}"
    )

    print(
        f"Image dtype: "
        f"{images.dtype}"
    )

    print(
        f"Image range: "
        f"{images.min().item():.3f} - "
        f"{images.max().item():.3f}"
    )

    print(
        f"Labels: "
        f"{labels[:10].tolist()}"
    )

    assert images.shape == (
        BATCH_SIZE,
        3,
        224,
        224,
    )

    assert labels.shape == (
        BATCH_SIZE,
    )

    assert images.dtype == torch.float32

    assert (
        images.min() >= 0
        and images.max() <= 1
    )

    print(
        "✓ DataLoader batch verified"
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    print("\nCreating DSVNet...")

    model = DSVNet(
        num_classes=NUM_CLASSES
    ).to(device)

    # ------------------------------------------------------
    # Loss + optimizer
    # ------------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ------------------------------------------------------
    # GPU memory reset
    # ------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    print("\nStarting smoke test...")

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        print(
            f"\nEpoch {epoch}/{MAX_EPOCHS}"
        )

        train_loss, train_acc = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Acc:  {train_acc * 100:.2f}%"
        )

        print(
            f"Val Loss:   {val_loss:.4f}"
        )

        print(
            f"Val Acc:    {val_acc * 100:.2f}%"
        )

    # ------------------------------------------------------
    # GPU memory
    # ------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.synchronize()

        peak = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(
            f"\nPeak GPU memory: "
            f"{peak:.3f} GB"
        )

    # ------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------

    checkpoint_dir = Path(
        "checkpoints"
    ) / "dsvnet"

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint_path = (
        checkpoint_dir
        / "smoke_test.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "epoch":
                MAX_EPOCHS,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_acc,

            "val_loss":
                val_loss,

            "val_accuracy":
                val_acc,
        },
        checkpoint_path
    )

    print(
        f"\nCheckpoint saved to:"
    )

    print(
        checkpoint_path.resolve()
    )

    print("\n" + "=" * 70)
    print("SMOKE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()