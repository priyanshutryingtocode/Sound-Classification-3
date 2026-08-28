import gc

import torch
import torch.nn as nn

from models.dsvnet import DSVNet


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZES = [1, 2, 4, 8, 16]

IMAGE_SIZE = 224
NUM_CLASSES = 10


def cleanup():
    """
    Clean up GPU memory between tests.
    """

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def benchmark_batch(batch_size):

    print("\n" + "-" * 60)
    print(f"Testing batch size: {batch_size}")
    print("-" * 60)

    cleanup()

    model = None
    optimizer = None

    try:

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        model = DSVNet(
            num_classes=NUM_CLASSES
        ).to(DEVICE)

        model.train()

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.0001
        )

        criterion = nn.CrossEntropyLoss()

        # --------------------------------------------------
        # Dummy batch
        # --------------------------------------------------

        x = torch.randn(
            batch_size,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
            device=DEVICE
        )

        y = torch.randint(
            0,
            NUM_CLASSES,
            (batch_size,),
            device=DEVICE
        )

        # --------------------------------------------------
        # Forward
        # --------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(x)

        loss = criterion(
            output,
            y
        )

        # --------------------------------------------------
        # Backward
        # --------------------------------------------------

        loss.backward()

        # --------------------------------------------------
        # Optimizer step
        # --------------------------------------------------

        optimizer.step()

        # --------------------------------------------------
        # Memory
        # --------------------------------------------------

        torch.cuda.synchronize()

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        peak = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        print(
            f"Loss:      {loss.item():.4f}"
        )

        print(
            f"Allocated: {allocated:.3f} GB"
        )

        print(
            f"Reserved:  {reserved:.3f} GB"
        )

        print(
            f"Peak:      {peak:.3f} GB"
        )

        print("✓ BATCH FITS")

        result = {
            "batch_size": batch_size,
            "success": True,
            "peak_gb": peak,
        }

    except torch.cuda.OutOfMemoryError:

        print("✗ CUDA OUT OF MEMORY")

        result = {
            "batch_size": batch_size,
            "success": False,
            "peak_gb": None,
        }

        # Clear the failed CUDA state.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except RuntimeError as e:

        if "out of memory" in str(e).lower():

            print("✗ CUDA OUT OF MEMORY")

            result = {
                "batch_size": batch_size,
                "success": False,
                "peak_gb": None,
            }

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        else:

            raise

    finally:

        del optimizer
        del model

        if "x" in locals():
            del x

        if "y" in locals():
            del y

        if "output" in locals():
            del output

        if "loss" in locals():
            del loss

        cleanup()

    return result


def main():

    print("=" * 60)
    print("DSVNet Training Memory Benchmark")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        total_memory = (
            torch.cuda.get_device_properties(0)
            .total_memory
            / 1024**3
        )

        print(
            f"VRAM: {total_memory:.2f} GB"
        )

    results = []

    for batch_size in BATCH_SIZES:

        result = benchmark_batch(
            batch_size
        )

        results.append(result)

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        f"{'Batch':<10}"
        f"{'Status':<15}"
        f"{'Peak VRAM':<15}"
    )

    print("-" * 40)

    for result in results:

        status = (
            "PASS"
            if result["success"]
            else "OOM"
        )

        peak = (
            f"{result['peak_gb']:.3f} GB"
            if result["peak_gb"] is not None
            else "-"
        )

        print(
            f"{result['batch_size']:<10}"
            f"{status:<15}"
            f"{peak:<15}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()