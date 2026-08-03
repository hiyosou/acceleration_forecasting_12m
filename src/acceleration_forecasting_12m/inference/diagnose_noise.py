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
    if name == "guide_disabled":
        for key in ("guide_values", "guide_deltas", "guide_mask", "guide_similarities", "retrieval_mask"):
            output[key] = torch.zeros_like(output[key])
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
    process, checkpoint_data = load_process(checkpoint, device)
    prediction_type = checkpoint_data.get("model_config", {}).get("prediction_type", "epsilon")
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
        normal_prediction = None
        for scenario in ("normal", "guide_shuffled", "guide_disabled"):
            predicted = process.model(noisy, timesteps, _scenario(batch, scenario))
            model_stats = process.model.diagnostic_stats() if hasattr(process.model, "diagnostic_stats") else {}
            if normal_prediction is None:
                normal_prediction = predicted.detach()
            mask = batch["target_mask"]
            expected = noise if prediction_type == "epsilon" else (
                alpha.sqrt() * noise - (1 - alpha).sqrt() * batch["target"]
            )
            prediction_mse = (((predicted - expected).square() * mask).sum() / mask.sum().clamp_min(1)).item()
            predicted_x0, _ = process.model_output_to_x0_epsilon(noisy, predicted, alpha)
            error = (predicted_x0 - batch["target"]) * float(dataset.residual_norm.std)
            valid = mask > 0
            rows.append({
                "timestep_start": lower, "timestep_end": upper, "scenario": scenario,
                "prediction_type": prediction_type,
                "prediction_mse": prediction_mse, "prediction_rmse": float(np.sqrt(prediction_mse)),
                "x0_mae_physical": float(error[valid].abs().mean()),
                "x0_rmse_physical": float(error[valid].square().mean().sqrt()),
                "mean_snr": float((alpha / (1 - alpha).clamp_min(1e-12)).mean()),
                "mean_valid_months": float(mask.sum(dim=1).mean()),
                "output_mae_from_normal": float((predicted - normal_prediction).abs().mean()),
                **model_stats,
            })
    frame = pd.DataFrame(rows)
    normal = frame.loc[frame["scenario"] == "normal", ["timestep_start", "prediction_mse"]].rename(columns={"prediction_mse": "normal_prediction_mse"})
    frame = frame.merge(normal, on="timestep_start", how="left")
    frame["prediction_mse_change_from_normal"] = frame["prediction_mse"] - frame["normal_prediction_mse"]
    frame.to_csv(output_dir / "timestep_condition_diagnostics.csv", index=False, encoding="utf-8-sig")
    # epsilon-MSE and v-MSE use different targets and are not compared directly.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [f"{a}-{b}" for a, b in bins]
    for scenario, group in frame.groupby("scenario", sort=False):
        group = group.sort_values("timestep_start")
        axes[0].plot(labels, group["prediction_mse"], marker="o", label=scenario)
        axes[1].plot(labels, group["x0_mae_physical"], marker="o", label=scenario)
    axes[0].set_ylabel(f"{'v' if prediction_type == 'v_prediction' else 'epsilon'} MSE")
    axes[1].set_ylabel("x0 MAE [m/s²]")
    for axis in axes: axis.set_xlabel("timestep"); axis.grid(color="0.85"); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output_dir / "timestep_condition_diagnostics.png", dpi=150); plt.close(fig)
    summary = {
        "validation_records": len(dataset), "checkpoint": str(Path(checkpoint).resolve()),
        "prediction_type": prediction_type,
        "mean_normal_prediction_mse": float(frame.loc[frame.scenario == "normal", "prediction_mse"].mean()),
        "mean_normal_x0_mae_physical": float(frame.loc[frame.scenario == "normal", "x0_mae_physical"].mean()),
        "device": str(device), "seed": int(seed),
    }
    write_json(output_dir / "diagnostic_summary.json", summary)
    return summary
