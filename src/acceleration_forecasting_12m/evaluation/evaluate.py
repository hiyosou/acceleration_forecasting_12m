from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from acceleration_forecasting_12m.common.io import write_json
from acceleration_forecasting_12m.common.progress import progress_bar
from .metrics import target_metrics
from .visualize import plot_target


def _sample_map(sample_dir, wanted):
    output = {}
    for path in sorted(Path(sample_dir).glob("samples_*.npz")):
        data = np.load(path)
        for target_id, samples in zip(data["target_ids"].astype(str), data["samples"]):
            if target_id in wanted: output[target_id] = samples
    return output


def evaluate(dataset_dir, prediction_dir, output_dir, *, bootstrap=1000, seed=42,
             plot=False, plot_max_targets=100, y_max=6.0, dpi=150, progress=True):
    dataset_dir, prediction_dir, output_dir = map(Path, (dataset_dir, prediction_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = dataset_dir / "inference" / "inputs"; targets = dataset_dir / "inference" / "targets"
    metadata = pd.read_csv(inputs / "metadata.csv", encoding="utf-8-sig")
    target_ids = pd.read_csv(targets / "target_ids.csv", encoding="utf-8-sig")["target_id"].astype(str)
    target_values, target_masks = np.load(targets / "target_values.npy"), np.load(targets / "target_masks.npy")
    target_lookup = {value: index for index, value in enumerate(target_ids)}
    predictions = pd.read_csv(prediction_dir / "predictions.csv", encoding="utf-8-sig")
    rows = []
    for target_id, group in predictions.groupby(predictions["target_id"].astype(str), sort=False):
        if target_id not in target_lookup: continue
        index = target_lookup[target_id]; ordered = group.sort_values("month_index")
        metrics = target_metrics(target_values[index], target_masks[index], ordered["prediction_median"],
                                 ordered["prediction_p10"], ordered["prediction_p90"])
        if metrics is None: continue
        meta = metadata.loc[metadata["target_id"].astype(str) == target_id].iloc[0]
        rows.append({"target_id": target_id, "dataset_id": meta["dataset_id"], "direction": meta["direction"],
                     "bin_start_m": meta["bin_start_m"], **metrics})
    frame = pd.DataFrame(rows); frame.to_csv(output_dir / "evaluation_per_target.csv", index=False, encoding="utf-8-sig")
    metric_names = ("MAE", "MSE", "RMSE", "correlation", "peak_value_error", "peak_month_error",
                    "coverage_p10_p90", "mean_interval_width", "median_adjacent_abs_difference")
    summary = {name: float(frame[name].mean(skipna=True)) for name in metric_names}
    rng = np.random.default_rng(seed); datasets = frame["dataset_id"].unique(); bootstrap_rows = []
    for _ in progress_bar(range(int(bootstrap)), enabled=progress, desc="dataset bootstrap", unit="iteration"):
        sampled = rng.choice(datasets, size=len(datasets), replace=True)
        parts = [frame.loc[frame["dataset_id"] == value] for value in sampled]
        combined = pd.concat(parts, ignore_index=True)
        bootstrap_rows.append({name: combined[name].mean(skipna=True) for name in metric_names})
    bootstrap_frame = pd.DataFrame(bootstrap_rows); bootstrap_frame.to_csv(output_dir / "bootstrap_results.csv", index=False)
    for name in metric_names:
        summary[f"{name}_ci95_low"], summary[f"{name}_ci95_high"] = map(
            float, np.nanpercentile(bootstrap_frame[name], [2.5, 97.5])
        )
    summary.update({"target_count": int(len(frame)), "bootstrap_iterations": int(bootstrap), "forecast_months": 12})
    write_json(output_dir / "evaluation_summary.json", summary)
    if plot and not frame.empty:
        selected = frame.sort_values("RMSE", ascending=False).head(int(plot_max_targets))["target_id"].astype(str).tolist()
        samples = _sample_map(prediction_dir / "samples", set(selected))
        history_values, history_masks = np.load(inputs / "history_values.npy"), np.load(inputs / "history_masks.npy")
        guide_values, guide_masks = np.load(inputs / "guide_values.npy"), np.load(inputs / "guide_masks.npy")
        baselines = np.load(inputs / "guide_baselines.npy")
        meta_lookup = {str(row.target_id): (index, row._asdict()) for index, row in enumerate(metadata.itertuples(index=False))}
        for target_id in progress_bar(selected, enabled=progress, desc="評価画像", unit="image"):
            if target_id not in samples: continue
            data_index, meta = meta_lookup[target_id]; target_index = target_lookup[target_id]
            plot_target(
                output_dir / "plots" / str(meta["direction"]) / f"{target_id}.png", meta,
                history_values[data_index], history_masks[data_index], guide_values[data_index],
                guide_masks[data_index], baselines[data_index],
                predictions.loc[predictions["target_id"].astype(str) == target_id],
                target_values[target_index], target_masks[target_index], samples[target_id], y_max=y_max, dpi=dpi,
            )
    return summary
