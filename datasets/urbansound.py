from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset


class UrbanSound8KDataset(Dataset):
    """
    UrbanSound8K dataset loader.

    Expected structure:

        UrbanSound8K/
        ├── audio/
        │   ├── fold1/
        │   ├── fold2/
        │   └── ...
        │       fold10/
        │
        └── metadata/
            └── UrbanSound8K.csv

    Each item returns:

        waveform : Tensor [channels, samples]
        label    : int
        metadata : dict
    """

    def __init__(self, dataset_root, folds=None):

        self.dataset_root = Path(dataset_root)

        self.audio_dir = self.dataset_root / "audio"
        self.metadata_dir = self.dataset_root / "metadata"

        # --------------------------------------------------
        # Locate metadata CSV
        # --------------------------------------------------

        csv_files = list(self.metadata_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV file found in {self.metadata_dir}"
            )

        if len(csv_files) > 1:
            raise RuntimeError(
                f"Multiple CSV files found: {csv_files}"
            )

        self.metadata_path = csv_files[0]

        # --------------------------------------------------
        # Load metadata
        # --------------------------------------------------

        self.metadata = pd.read_csv(
            self.metadata_path
        )

        required_columns = {
            "slice_file_name",
            "fold",
            "classID",
            "class",
        }

        missing = required_columns - set(
            self.metadata.columns
        )

        if missing:
            raise ValueError(
                f"Missing metadata columns: {missing}"
            )

        # --------------------------------------------------
        # Filter folds
        # --------------------------------------------------

        if folds is not None:

            if isinstance(folds, int):
                folds = [folds]

            folds = list(folds)

            invalid_folds = [
                fold
                for fold in folds
                if fold < 1 or fold > 10
            ]

            if invalid_folds:
                raise ValueError(
                    f"Invalid folds: {invalid_folds}"
                )

            self.metadata = self.metadata[
                self.metadata["fold"].isin(folds)
            ].reset_index(drop=True)

        # --------------------------------------------------
        # Resolve audio paths
        # --------------------------------------------------

        self.file_paths = []

        for _, row in self.metadata.iterrows():

            fold = int(row["fold"])
            filename = row["slice_file_name"]

            path = (
                self.audio_dir
                / f"fold{fold}"
                / filename
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Audio file not found: {path}"
                )

            self.file_paths.append(path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        audio_path = self.file_paths[index]

        # --------------------------------------------------
        # Load WAV using soundfile
        # --------------------------------------------------

        waveform, sample_rate = sf.read(
            str(audio_path),
            dtype="float32"
        )

        # soundfile returns:
        #
        # mono   -> [samples]
        # stereo -> [samples, channels]
        #
        # We want:
        #
        # [channels, samples]

        if waveform.ndim == 1:

            waveform = waveform[None, :]

        else:

            waveform = waveform.T

        waveform = torch.from_numpy(waveform)

        label = int(row["classID"])

        metadata = {
            "filename": row["slice_file_name"],
            "fold": int(row["fold"]),
            "class_id": label,
            "class_name": row["class"],
            "sample_rate": sample_rate,
            "path": str(audio_path),
        }

        return waveform, label, metadata