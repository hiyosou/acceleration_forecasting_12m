from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from acceleration_forecasting_12m.common.io import write_json
from acceleration_forecasting_12m.common.progress import progress_bar
from acceleration_forecasting_12m.datasets.torch_dataset import ForecastDataset
from acceleration_forecasting_12m.evaluation.metrics import target_metrics
from .sampling import load_process, sample_target


def validate(dataset_dir, checkpoint, output_dir, *, device=None, num_samples=100,
             sampling_steps=50, max_records=None, seed=42, progress=True,
             initial_noise_scale=1.0, cfg_scale=1.0, variance_scale=1.0):
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    process, _ = load_process(checkpoint, device)
    dataset = ForecastDataset(dataset_dir / "model_validation", dataset_dir)
    count = min(len(dataset), int(max_records)) if max_records is not None else len(dataset)
    rows = []
    for index in progress_bar(range(count), enabled=progress, total=count, desc="validation生成", unit="target"):
        meta = dataset.metadata.iloc[index]; target_id = str(meta["target_id"])
        samples = sample_target(process, dataset, index, target_id, num_samples=num_samples,
                                sampling_steps=sampling_steps, device=device, seed=seed,
                                initial_noise_scale=initial_noise_scale, cfg_scale=cfg_scale,
                                variance_scale=variance_scale)
        median = np.median(samples, axis=0); p10 = np.percentile(samples, 10, axis=0); p90 = np.percentile(samples, 90, axis=0)
        metrics = target_metrics(dataset.physical_target(index), dataset.target_masks[index], median, p10, p90)
        if metrics is not None:
            rows.append({"target_id": target_id, "dataset_id": meta["dataset_id"],
                         "ensemble_mean_std": float(np.mean(np.std(samples, axis=0))), **metrics})
    frame = pd.DataFrame(rows); frame.to_csv(output_dir / "validation_per_target.csv", index=False, encoding="utf-8-sig")
    summary = {column: float(frame[column].mean(skipna=True)) for column in (
        "MAE", "MSE", "RMSE", "correlation", "peak_value_error", "peak_month_error",
        "coverage_p10_p90", "mean_interval_width", "median_adjacent_abs_difference",
        "ensemble_mean_std",
    )}
    summary.update({
        "target_count": int(len(frame)), "all_finite": bool(np.isfinite(frame[["MAE", "RMSE", "mean_interval_width"]]).all().all()),
        "forecast_months": 12, "num_samples": int(num_samples), "sampling_steps": int(sampling_steps),
        "initial_noise_scale": float(initial_noise_scale),
        "cfg_scale": float(cfg_scale), "variance_scale": float(variance_scale),
    })
    write_json(output_dir / "validation_summary.json", summary); return summary
