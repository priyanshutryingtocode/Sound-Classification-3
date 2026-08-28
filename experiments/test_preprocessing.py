from pathlib import Path

import matplotlib.pyplot as plt

from preprocessing.representations import (
    load_audio,
    compute_mel,
    compute_stft,
    compute_cqt,
    compute_chromagram
)

from preprocessing.fusion import (
    resize_feature,
    create_rgb_fusion
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATASET_ROOT = Path(
    r"D:\BTP\Dataset\UrbanSound8K"
)

AUDIO_PATH = (
    DATASET_ROOT
    / "audio"
    / "fold5"
    / "100032-3-0-0.wav"
)

OUTPUT_DIR = Path(
    "results"
) / "preprocessing_test"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    print("=" * 60)
    print("DSVNet Preprocessing Test")
    print("=" * 60)

    print(f"\nAudio:")
    print(AUDIO_PATH)

    # --------------------------------------------------
    # Load audio
    # --------------------------------------------------

    y, sr = load_audio(
        str(AUDIO_PATH)
    )

    print("\nAudio information:")
    print(f"Sample rate: {sr}")
    print(f"Samples:     {len(y)}")
    print(
        f"Duration:    {len(y) / sr:.3f} seconds"
    )

    # --------------------------------------------------
    # Generate representations
    # --------------------------------------------------

    print("\nGenerating representations...")

    mel = compute_mel(y, sr)

    print(
        f"MEL shape:   {mel.shape}"
    )

    stft = compute_stft(y, sr)

    print(
        f"STFT shape:  {stft.shape}"
    )

    cqt = compute_cqt(y, sr)

    print(
        f"CQT shape:   {cqt.shape}"
    )

    chroma = compute_chromagram(y, sr)

    print(
        f"Chroma shape:{chroma.shape}"
    )

    # --------------------------------------------------
    # Resize individual representations
    # --------------------------------------------------

    mel_img = resize_feature(mel)
    stft_img = resize_feature(stft)
    cqt_img = resize_feature(cqt)
    chroma_img = resize_feature(chroma)

    print("\nResized representations:")

    print(
        f"MEL:     {mel_img.shape}"
    )

    print(
        f"STFT:    {stft_img.shape}"
    )

    print(
        f"CQT:     {cqt_img.shape}"
    )

    print(
        f"Chroma:  {chroma_img.shape}"
    )

    # --------------------------------------------------
    # RGB fusion
    # --------------------------------------------------

    rgb = create_rgb_fusion(
        mel,
        stft,
        cqt
    )

    print(
        f"\nRGB fusion shape: {rgb.shape}"
    )

    print(
        f"RGB dtype:        {rgb.dtype}"
    )

    print(
        f"RGB range:        "
        f"{rgb.min()} - {rgb.max()}"
    )

    assert rgb.shape == (
        224,
        224,
        3
    )

    assert rgb.dtype.name == "uint8"

    print(
        "✓ RGB fusion format verified"
    )

    # --------------------------------------------------
    # Save individual representations
    # --------------------------------------------------

    plt.figure(figsize=(8, 4))
    plt.imshow(
        mel_img,
        aspect="auto",
        origin="lower",
        cmap="gray"
    )
    plt.title("MEL")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "mel.png",
        dpi=150
    )
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.imshow(
        stft_img,
        aspect="auto",
        origin="lower",
        cmap="gray"
    )
    plt.title("STFT")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "stft.png",
        dpi=150
    )
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.imshow(
        cqt_img,
        aspect="auto",
        origin="lower",
        cmap="gray"
    )
    plt.title("CQT")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "cqt.png",
        dpi=150
    )
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.imshow(
        chroma_img,
        aspect="auto",
        origin="lower",
        cmap="gray"
    )
    plt.title("Chromagram")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "chroma.png",
        dpi=150
    )
    plt.close()

    # --------------------------------------------------
    # Save fused RGB
    # --------------------------------------------------

    plt.figure(figsize=(8, 4))
    plt.imshow(rgb)
    plt.title(
        "MEL + STFT + CQT RGB Fusion"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "fusion_rgb.png",
        dpi=150
    )
    plt.close()

    print(
        f"\nImages saved to:"
    )
    print(
        OUTPUT_DIR.resolve()
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()