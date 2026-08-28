import torch

from models.dsvnet import DSVNet


def main():
    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("DSVNet Architecture Test")
    print("=" * 60)

    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = DSVNet(num_classes=10)
    model = model.to(device)

    print("\nModel created successfully.")

    # --------------------------------------------------
    # Parameter count
    # --------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # --------------------------------------------------
    # Dummy input
    # --------------------------------------------------

    x = torch.randn(
        1,
        3,
        224,
        224,
        device=device
    )

    print(f"\nInput shape: {tuple(x.shape)}")

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------

    model.train()

    output = model(x)

    print(f"Output shape: {tuple(output.shape)}")

    assert output.shape == (1, 10), \
        f"Unexpected output shape: {output.shape}"

    print("✓ Forward pass successful")

    # --------------------------------------------------
    # Backward pass
    # --------------------------------------------------

    target = torch.tensor(
        [3],
        dtype=torch.long,
        device=device
    )

    criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(output, target)

    print(f"Loss: {loss.item():.6f}")

    loss.backward()

    print("✓ Backward pass successful")

    # --------------------------------------------------
    # GPU memory
    # --------------------------------------------------

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated(device)
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved(device)
            / 1024**3
        )

        peak = (
            torch.cuda.max_memory_allocated(device)
            / 1024**3
        )

        print("\nGPU memory:")
        print(f"Allocated: {allocated:.3f} GB")
        print(f"Reserved:  {reserved:.3f} GB")
        print(f"Peak:      {peak:.3f} GB")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()