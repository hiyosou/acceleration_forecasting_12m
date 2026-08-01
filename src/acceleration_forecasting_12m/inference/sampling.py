from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from acceleration_forecasting_12m.datasets.torch_dataset import ForecastDataset
from acceleration_forecasting_12m.datasets.normalization import Normalization
from acceleration_forecasting_12m.diffusion.process import DiffusionProcess
from acceleration_forecasting_12m.models.unet import ResidualUNet12


def load_process(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    if config.get("cross_attention") is not False or config.get("forecast_months") != 12:
        raise ValueError("Checkpoint is not an attention-free twelve-month model")
    model = ResidualUNet12(config["dropout"]); model.load_state_dict(checkpoint["ema_state_dict"]); model.eval()
    return DiffusionProcess(model, 1000).to(device), checkpoint


def deterministic_seed(target_id, sample_offset=0, base_seed=42):
    digest = hashlib.sha256(f"{base_seed}:{target_id}:{sample_offset}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def repeat_condition(item, count, device):
    output = {}
    for key in ("current", "history", "history_mask"):
        value = item[key].to(device)
        output[key] = value.unsqueeze(0).repeat(count, *([1] * value.ndim))
    return output


def normalized_clip(dataset_root):
    config = json.loads((Path(dataset_root) / "guide_search_config.json").read_text(encoding="utf-8"))
    normalization = Normalization.load(Path(dataset_root) / "normalization.json")
    return tuple(float(value) for value in normalization.normalize(config["residual_clip_physical"]))


def sample_target(process, dataset, index, target_id, *, num_samples=100,
                  sampling_steps=50, device, seed=42):
    item = dataset[index]; batch = repeat_condition(item, int(num_samples), device)
    generator = torch.Generator(device=device).manual_seed(deterministic_seed(target_id, 0, seed))
    generated = process.ddim(
        batch, shape=(int(num_samples), 12), sampling_steps=sampling_steps, eta=0,
        normalized_clip=normalized_clip(dataset.root), generator=generator,
    ).cpu().numpy()
    return dataset.physical_prediction(generated, index)

