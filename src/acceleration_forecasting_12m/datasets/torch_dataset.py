from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .normalization import Normalization
from acceleration_forecasting_12m.common.constants import PHYSICAL_MAX, PHYSICAL_MIN


INPUT_ARRAYS = (
    "current_values", "history_values", "history_masks", "guide_values", "guide_masks",
    "guide_similarities", "retrieval_masks", "guide_baselines", "guide_softmax_weights",
)


class ForecastDataset(Dataset):
    def __init__(self, split_dir, dataset_root, *, include_targets=True):
        self.path, self.root = Path(split_dir), Path(dataset_root)
        self.metadata = pd.read_csv(self.path / "metadata.csv", encoding="utf-8-sig")
        self.arrays = {name: np.load(self.path / f"{name}.npy", mmap_mode="r") for name in INPUT_ARRAYS}
        self.include_targets = bool(include_targets)
        if include_targets:
            self.targets = np.load(self.path / "target_values.npy", mmap_mode="r")
            self.target_masks = np.load(self.path / "target_masks.npy", mmap_mode="r")
        self.residual_norm = Normalization.load(self.root / "normalization.json")
        self.condition_norm = Normalization.load(self.root / "condition_normalization.json")

    def __len__(self):
        return len(self.metadata)

    @staticmethod
    def _finite_zero(values):
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def __getitem__(self, index):
        current = self._finite_zero(self.condition_norm.normalize(self.arrays["current_values"][index]))
        history = self._finite_zero(self.condition_norm.normalize(self.arrays["history_values"][index]))
        item = {
            "current": torch.from_numpy(current.copy()), "history": torch.from_numpy(history.copy()),
            "history_mask": torch.from_numpy(self.arrays["history_masks"][index].copy().astype(np.float32)),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.include_targets:
            target = self._finite_zero(self.residual_norm.normalize(self.targets[index]))
            item["target"] = torch.from_numpy(target.copy())
            item["target_mask"] = torch.from_numpy(self.target_masks[index].copy().astype(np.float32))
        return item

    def denormalize_residual(self, values):
        return self.residual_norm.denormalize(values)

    def physical_prediction(self, residual_normalized, index, bounds=(PHYSICAL_MIN, PHYSICAL_MAX)):
        residual = self.denormalize_residual(residual_normalized)
        baseline = np.asarray(self.arrays["guide_baselines"][index], dtype=np.float32)
        return np.clip(residual + baseline, float(bounds[0]), float(bounds[1]))

    def physical_target(self, index):
        if not self.include_targets:
            raise ValueError("Targets are not loaded")
        return np.asarray(self.targets[index], dtype=np.float32) + np.asarray(
            self.arrays["guide_baselines"][index], dtype=np.float32
        )
