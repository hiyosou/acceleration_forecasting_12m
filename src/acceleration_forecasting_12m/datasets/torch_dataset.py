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
OPTIONAL_ARRAYS = ("guide_deltas",)


class ForecastDataset(Dataset):
    def __init__(self, split_dir, dataset_root, *, include_targets=True):
        self.path, self.root = Path(split_dir), Path(dataset_root)
        self.metadata = pd.read_csv(self.path / "metadata.csv", encoding="utf-8-sig")
        self.arrays = {name: np.load(self.path / f"{name}.npy", mmap_mode="r") for name in INPUT_ARRAYS}
        for name in OPTIONAL_ARRAYS:
            file = self.path / f"{name}.npy"
            if file.is_file():
                self.arrays[name] = np.load(file, mmap_mode="r")
        self.include_targets = bool(include_targets)
        if include_targets:
            self.targets = np.load(self.path / "target_values.npy", mmap_mode="r")
            self.target_masks = np.load(self.path / "target_masks.npy", mmap_mode="r")
        self.residual_norm = Normalization.load(self.root / "normalization.json")
        self.condition_norm = Normalization.load(self.root / "condition_normalization.json")
        config_path = self.root / "guide_search_config.json"
        self.config = __import__("json").loads(config_path.read_text(encoding="utf-8"))
        self.target_mode = self.config.get("target_mode", "residual")

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
        guide_values = self._finite_zero(self.condition_norm.normalize(self.arrays["guide_values"][index]))
        raw_deltas = self.arrays.get("guide_deltas")
        guide_deltas = np.zeros_like(guide_values) if raw_deltas is None else self._finite_zero(
            np.asarray(raw_deltas[index]) / self.condition_norm.std
        )
        item.update({
            "guide_values": torch.from_numpy(guide_values.copy()),
            "guide_deltas": torch.from_numpy(guide_deltas.copy()),
            "guide_mask": torch.from_numpy(self.arrays["guide_masks"][index].copy().astype(np.float32)),
            "guide_similarities": torch.from_numpy(self.arrays["guide_similarities"][index].copy().astype(np.float32)),
            "retrieval_mask": torch.from_numpy(self.arrays["retrieval_masks"][index].copy().astype(np.float32)),
        })
        if self.include_targets:
            target = self._finite_zero(self.residual_norm.normalize(self.targets[index]))
            item["target"] = torch.from_numpy(target.copy())
            item["target_mask"] = torch.from_numpy(self.target_masks[index].copy().astype(np.float32))
        return item

    def denormalize_residual(self, values):
        return self.residual_norm.denormalize(values)

    def physical_prediction(self, residual_normalized, index, bounds=(PHYSICAL_MIN, PHYSICAL_MAX)):
        values = self.denormalize_residual(residual_normalized)
        if self.target_mode == "residual":
            values = values + np.asarray(self.arrays["guide_baselines"][index], dtype=np.float32)
        return np.clip(values, float(bounds[0]), float(bounds[1]))

    def physical_target(self, index):
        if not self.include_targets:
            raise ValueError("Targets are not loaded")
        values = np.asarray(self.targets[index], dtype=np.float32)
        if self.target_mode == "residual":
            values = values + np.asarray(self.arrays["guide_baselines"][index], dtype=np.float32)
        return values
