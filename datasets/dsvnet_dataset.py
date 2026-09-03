from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class DSVNetDataset(Dataset):
    """
    Dataset for preprocessed DSVNet representations.

    representation:
        "rgb"  -> original MEL + STFT + CQT
        "mel"  -> MEL channel only, replicated to 3 channels
        "stft" -> STFT channel only, replicated to 3 channels
        "cqt"  -> CQT channel only, replicated to 3 channels

    The replication is necessary because the verified DSVNet
    architecture expects a 3-channel input.
    """

    def __init__(
        self,
        manifest_path,
        folds,
        transform=None,
        representation="rgb",
    ):
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.representation = representation.lower()

        valid_representations = {
            "rgb",
            "mel",
            "stft",
            "cqt",
        }

        if self.representation not in valid_representations:
            raise ValueError(
                f"Invalid representation: {representation}. "
                f"Choose from {valid_representations}"
            )

        if isinstance(folds, int):
            folds = [folds]

        folds = list(folds)

        # --------------------------------------------------
        # Load manifest
        # --------------------------------------------------

        self.metadata = pd.read_csv(
            self.manifest_path
        )

        required_columns = {
            "slice_file_name",
            "fold",
            "classID",
            "class",
            "processed_path",
        }

        missing = (
            required_columns
            - set(self.metadata.columns)
        )

        if missing:
            raise ValueError(
                f"Missing manifest columns: {missing}"
            )

        # --------------------------------------------------
        # Filter folds
        # --------------------------------------------------

        self.metadata = self.metadata[
            self.metadata["fold"].isin(folds)
        ].reset_index(drop=True)

        if len(self.metadata) == 0:
            raise ValueError(
                f"No samples found for folds: {folds}"
            )

        # --------------------------------------------------
        # Validate processed files
        # --------------------------------------------------

        self.file_paths = []

        for _, row in self.metadata.iterrows():

            path = Path(
                row["processed_path"]
            )

            if not path.is_absolute():
                path = Path.cwd() / path

            if not path.exists():
                raise FileNotFoundError(
                    f"Processed image not found: {path}"
                )

            self.file_paths.append(path)

        self.folds = folds

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        image_path = self.file_paths[index]

        # --------------------------------------------------
        # Load RGB image
        # --------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # --------------------------------------------------
        # Select representation
        # --------------------------------------------------

        if self.representation == "rgb":

            # Keep original:
            # R = MEL
            # G = STFT
            # B = CQT

            image_array = torch.from_numpy(
                __import__("numpy").asarray(
                    image,
                    dtype=__import__("numpy").float32,
                )
            )

        else:

            import numpy as np

            image_array = np.asarray(
                image,
                dtype=np.float32,
            )

            if self.representation == "mel":
                channel = image_array[:, :, 0]

            elif self.representation == "stft":
                channel = image_array[:, :, 1]

            elif self.representation == "cqt":
                channel = image_array[:, :, 2]

            # Replicate the selected representation
            # into all three channels.
            image_array = np.stack(
                [channel, channel, channel],
                axis=2,
            )

            image_array = torch.from_numpy(
                image_array
            )

        # --------------------------------------------------
        # HWC -> CHW
        # --------------------------------------------------

        image = (
            image_array
            .permute(2, 0, 1)
            / 255.0
        )

        image = image.float()

        # --------------------------------------------------
        # Label
        # --------------------------------------------------

        label = int(
            row["classID"]
        )

        return image, label