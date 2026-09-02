from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class DSVNetDataset(Dataset):
    """
    Dataset for the preprocessed DSVNet RGB images.

    Each sample:
        PNG -> RGB -> Tensor [3, 224, 224]

    The manifest preserves the official UrbanSound8K fold
    and class assignments.
    """

    def __init__(
        self,
        manifest_path,
        folds,
        transform=None,
    ):
        self.manifest_path = Path(manifest_path)
        self.transform = transform

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

            # Convert relative paths to project-relative
            # paths if necessary.
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
        # Transform
        # --------------------------------------------------

        if self.transform is not None:

            image = self.transform(
                image
            )

        else:

            # Manual PIL -> Tensor conversion.
            #
            # [H, W, C]
            #      ↓
            # [C, H, W]
            #      ↓
            # float32 [0, 1]

            import numpy as np

            image = np.asarray(
                image,
                dtype=np.float32
            )

            image = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                / 255.0
            )

        label = int(
            row["classID"]
        )

        return image, label