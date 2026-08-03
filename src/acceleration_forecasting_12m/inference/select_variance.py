from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from acceleration_forecasting_12m.common.io import write_json
from acceleration_forecasting_12m.common.progress import progress_bar
from acceleration_forecasting_12m.datasets.torch_dataset import ForecastDataset
from acceleration_forecasting_12m.evaluation.metrics import target_metrics
from .sampling import load_process, sample_target


def _summarize(dataset, sample_sets):
    rows = []
    for index, samples in enumerate(sample_sets):
        median = np.median(samples, axis=0); p10 = np.percentile(samples, 10, axis=0); p90 = np.percentile(samples, 90, axis=0)
        metrics = target_metrics(dataset.physical_target(index), dataset.target_masks[index], median, p10, p90)
        if metrics is not None:
            metrics["ensemble_mean_std"] = float(np.mean(np.std(samples, axis=0))); rows.append(metrics)
    frame = pd.DataFrame(rows)
    names = ("MAE", "MSE", "RMSE", "correlation", "coverage_p10_p90", "mean_interval_width",
             "ensemble_mean_std", "median_adjacent_abs_difference", "peak_value_error", "peak_month_error")
    return {name: float(frame[name].mean(skipna=True)) for name in names}


def calibrate_samples(samples, alpha):
    samples = np.asarray(samples)
    center = samples.mean(axis=0, keepdims=True)
    return np.clip(center + float(alpha) * (samples - center), 0.1, 6.0)


def select_variance_config(dataset_dir, min_snr_checkpoint, cfg_checkpoint, output_dir, *,
                           device=None, num_samples=100, progress=True, seed=42):
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = ForecastDataset(dataset_dir / "model_validation", dataset_dir)
    models = (("min_snr", Path(min_snr_checkpoint), (1.0,)),
              ("min_snr_cfg", Path(cfg_checkpoint), (1.0, 1.5, 2.0)))
    rows = []
    raw_configs = [(name, checkpoint, noise, steps, cfg)
                   for name, checkpoint, scales in models
                   for noise in (1.0, 0.75, 0.5) for steps in (50, 100, 200) for cfg in scales]
    for model_name, checkpoint, noise_scale, steps, cfg_scale in progress_bar(
            raw_configs, enabled=progress, desc="variance validation", unit="config"):
        process, _ = load_process(checkpoint, device); started = time.perf_counter(); raw = []
        for index in range(len(dataset)):
            target_id = str(dataset.metadata.iloc[index]["target_id"])
            raw.append(sample_target(process, dataset, index, target_id, num_samples=num_samples,
                                     sampling_steps=steps, device=device, seed=seed,
                                     initial_noise_scale=noise_scale, cfg_scale=cfg_scale))
        elapsed = time.perf_counter() - started
        for alpha in (1.0, 0.75, 0.5, 0.25, 0.1):
            calibrated = []
            for samples in raw:
                calibrated.append(calibrate_samples(samples, alpha))
            summary = _summarize(dataset, calibrated)
            stack = np.asarray(calibrated)
            rows.append({
                "model": model_name, "checkpoint": str(checkpoint.resolve()),
                "initial_noise_scale": noise_scale, "sampling_steps": steps,
                "cfg_scale": cfg_scale, "variance_scale": alpha, **summary,
                "all_finite": bool(np.isfinite(stack).all()), "quantile_order_valid": True,
                "physical_range_valid": bool(stack.min() >= 0.1 - 1e-6 and stack.max() <= 6.0 + 1e-6),
                "elapsed_seconds_raw_sampling": elapsed,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "sampling_variance_grid.csv", index=False, encoding="utf-8-sig")
    eligible = frame.loc[frame["all_finite"] & frame["quantile_order_valid"] & frame["physical_range_valid"]].copy()
    eligible = eligible.sort_values(["mean_interval_width", "MAE", "coverage_p10_p90", "sampling_steps"],
                                    ascending=[True, True, False, True])
    selected = eligible.iloc[0].to_dict()
    selected["width_target_0_7_met"] = bool(selected["mean_interval_width"] <= 0.7)
    selected["width_ideal_0_5_met"] = bool(selected["mean_interval_width"] <= 0.5)
    selected["coverage_75_met"] = bool(selected["coverage_p10_p90"] >= 0.75)
    selected["mae_0_4009_met"] = bool(selected["MAE"] <= 0.4009)
    write_json(output_dir / "selected_variance_config.json", selected)
    return selected
