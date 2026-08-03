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
from acceleration_forecasting_12m.models.absolute_attention_unet import AbsoluteAttentionUNet12
from acceleration_forecasting_12m.models.reference_modulated_unet import ReferenceModulatedUNet12


def load_process(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["model_config"]
    if config.get("forecast_months") != 12:
        raise ValueError("Checkpoint is not a twelve-month model")
    if config.get("attention_type") == "reference_modulated":
        model = ReferenceModulatedUNet12(config["dropout"])
    elif config.get("target_mode") == "absolute":
        model = AbsoluteAttentionUNet12(
            config["dropout"], condition_indicator=config.get("condition_indicator", False)
        )
    else:
        model = ResidualUNet12(config["dropout"])
    model.load_state_dict(checkpoint["ema_state_dict"]); model.eval()
    return DiffusionProcess(model, 1000, min_snr_gamma=config.get("min_snr_gamma")).to(device), checkpoint


def deterministic_seed(target_id, sample_offset=0, base_seed=42):
    digest = hashlib.sha256(f"{base_seed}:{target_id}:{sample_offset}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def repeat_condition(item, count, device):
    output = {}
    for key in ("current", "history", "history_mask", "guide_values", "guide_deltas",
                "guide_mask", "guide_similarities", "retrieval_mask"):
        value = item[key].to(device)
        output[key] = value.unsqueeze(0).repeat(count, *([1] * value.ndim))
    output["condition_present"] = torch.ones((count, 1), device=device)
    return output


def normalized_clip(dataset_root):
    config = json.loads((Path(dataset_root) / "guide_search_config.json").read_text(encoding="utf-8"))
    normalization = Normalization.load(Path(dataset_root) / "normalization.json")
    physical = config.get("sampling_clip_physical", config.get("residual_clip_physical"))
    return tuple(float(value) for value in normalization.normalize(physical))


def sample_target(process, dataset, index, target_id, *, num_samples=100,
                  sampling_steps=50, device, seed=42, initial_noise_scale=1.0, cfg_scale=1.0,
                  variance_scale=1.0):
    item = dataset[index]; batch = repeat_condition(item, int(num_samples), device)
    generator = torch.Generator(device=device).manual_seed(deterministic_seed(target_id, 0, seed))
    generated = process.ddim(
        batch, shape=(int(num_samples), 12), sampling_steps=sampling_steps, eta=0,
        normalized_clip=normalized_clip(dataset.root), generator=generator,
        initial_noise_scale=initial_noise_scale,
        cfg_scale=cfg_scale,
    ).cpu().numpy()
    physical = dataset.physical_prediction(generated, index)
    if float(variance_scale) != 1.0:
        center = physical.mean(axis=0, keepdims=True)
        physical = np.clip(center + float(variance_scale) * (physical - center), 0.1, 6.0)
    return physical
