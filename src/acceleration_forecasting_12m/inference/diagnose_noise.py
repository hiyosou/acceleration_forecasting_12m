from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from acceleration_forecasting_12m.common.io import write_json
from acceleration_forecasting_12m.datasets.torch_dataset import ForecastDataset
from .sampling import load_process


CONDITION_KEYS = ("current", "history", "history_mask", "guide_values", "guide_deltas",
                  "guide_mask", "guide_similarities", "retrieval_mask")


def _move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _scenario(batch, name):
    output = dict(batch)
    if name == "guide_shuffled":
        for key in ("guide_values", "guide_deltas", "guide_mask", "guide_similarities", "retrieval_mask"):
            output[key] = torch.roll(output[key], shifts=1, dims=0)
    if name in {"history_disabled", "all_conditions_disabled"}:
        for key in ("history", "history_mask"):
            output[key] = torch.zeros_like(output[key])
    if name == "all_conditions_disabled":
        for key in CONDITION_KEYS:
            output[key] = torch.zeros_like(output[key])
    return output


@torch.inference_mode()
def diagnose_noise(dataset_dir, checkpoint, output_dir, *, device=None, seed=42):
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    process, _ = load_process(checkpoint, device)
    dataset = ForecastDataset(dataset_dir / "model_validation", dataset_dir)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    batch = _move(next(iter(loader)), device)
    bins = [(0, 99), (100, 299), (300, 599), (600, 799), (800, 999)]
    rows, generator = [], torch.Generator(device=device).manual_seed(seed)
    for lower, upper in bins:
        timesteps = torch.randint(lower, upper + 1, (len(dataset),), device=device, generator=generator)
        noise = torch.randn(batch["target"].shape, device=device, generator=generator)
        alpha = process.alpha_bars[timesteps][:, None]
        noisy = alpha.sqrt() * batch["target"] + (1 - alpha).sqrt() * noise
        for scenario in ("normal", "guide_shuffled", "history_disabled", "all_conditions_disabled"):
            predicted = process.model(noisy, timesteps, _scenario(batch, scenario))
            mask = batch["target_mask"]
            epsilon_mse = (((predicted - noise).square() * mask).sum() / mask.sum().clamp_min(1)).item()
            predicted_x0 = (noisy - (1 - alpha).sqrt() * predicted) / alpha.sqrt()
            error = (predicted_x0 - batch["target"]) * float(dataset.residual_norm.std)
            valid = mask > 0
            rows.append({
                "timestep_start": lower, "timestep_end": upper, "scenario": scenario,
                "epsilon_mse": epsilon_mse, "epsilon_rmse": float(np.sqrt(epsilon_mse)),
                "x0_mae_physical": float(error[valid].abs().mean()),
                "x0_rmse_physical": float(error[valid].square().mean().sqrt()),
                "mean_snr": float((alpha / (1 - alpha).clamp_min(1e-12)).mean()),
                "mean_valid_months": float(mask.sum(dim=1).mean()),
            })
    frame = pd.DataFrame(rows)
    normal = frame.loc[frame["scenario"] == "normal", ["timestep_start", "epsilon_mse"]].rename(columns={"epsilon_mse": "normal_epsilon_mse"})
    frame = frame.merge(normal, on="timestep_start", how="left")
    frame["epsilon_mse_change_from_normal"] = frame["epsilon_mse"] - frame["normal_epsilon_mse"]
    frame.to_csv(output_dir / "timestep_condition_diagnostics.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [f"{a}-{b}" for a, b in bins]
    for scenario, group in frame.groupby("scenario", sort=False):
        group = group.sort_values("timestep_start")
        axes[0].plot(labels, group["epsilon_mse"], marker="o", label=scenario)
        axes[1].plot(labels, group["x0_mae_physical"], marker="o", label=scenario)
    axes[0].set_ylabel("epsilon MSE"); axes[1].set_ylabel("x0 MAE [m/s²]")
    for axis in axes: axis.set_xlabel("timestep"); axis.grid(color="0.85"); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output_dir / "timestep_condition_diagnostics.png", dpi=150); plt.close(fig)
    summary = {
        "validation_records": len(dataset), "checkpoint": str(Path(checkpoint).resolve()),
        "mean_normal_epsilon_mse": float(frame.loc[frame.scenario == "normal", "epsilon_mse"].mean()),
        "mean_normal_x0_mae_physical": float(frame.loc[frame.scenario == "normal", "x0_mae_physical"].mean()),
        "device": str(device), "seed": int(seed),
    }
    write_json(output_dir / "diagnostic_summary.json", summary)
    return summary
