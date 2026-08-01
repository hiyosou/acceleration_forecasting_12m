from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np


@dataclass(frozen=True)
class Normalization:
    mean: float
    std: float
    count: int
    source: str

    @classmethod
    def fit(cls, values, source):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < 2:
            raise ValueError("At least two finite values are required for normalization")
        std = float(values.std(ddof=0))
        if not np.isfinite(std) or std <= 1e-12:
            raise ValueError("Normalization standard deviation must be positive")
        return cls(float(values.mean()), std, int(values.size), str(source))

    def normalize(self, values):
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def denormalize(self, values):
        return np.asarray(values, dtype=np.float32) * self.std + self.mean

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

