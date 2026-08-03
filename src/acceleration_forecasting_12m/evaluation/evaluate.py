from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from acceleration_forecasting_12m.common.io import write_json
from acceleration_forecasting_12m.common.progress import progress_bar
from .metrics import target_metrics
from .visualize import plot_target


def _actual_history(dataset_dir):
    import json
    source = json.loads((Path(dataset_dir) / "source_artifacts.json").read_text(encoding="utf-8"))
    source_dir = Path(source["source_artifact_dir"])
    development = pd.read_csv(source_dir / "development_trends.csv", encoding="utf-8-sig")
    inference = pd.read_csv(source_dir / "inference_targets.csv", encoding="utf-8-sig").rename(columns={"target_id": "trend_id"})
    trends = pd.concat([development, inference], ignore_index=True)
    manifest = pd.read_csv(source_dir / "dataset_split_manifest.csv", encoding="utf-8-sig")
    speed = manifest.groupby("trend_id", as_index=False)["mean_velocity_kmh"].mean().rename(columns={"mean_velocity_kmh": "velocity"})
    trends = trends.merge(speed, on="trend_id", how="left")
    trends["measurement_date"] = pd.to_datetime(trends["measurement_date"], errors="coerce")
    trends["current_acc_z_max"] = pd.to_numeric(trends["current_acc_z_max"], errors="coerce")
    return trends


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
    residual_summary = output_dir.parent / "evaluation" / "evaluation_summary.json"
    if residual_summary.is_file() and residual_summary.resolve() != (output_dir / "evaluation_summary.json").resolve():
        import json
        previous = json.loads(residual_summary.read_text(encoding="utf-8"))
        comparison = [{"metric": name, "residual": previous[name], "absolute_attention": summary[name],
                       "difference": summary[name] - previous[name]}
                      for name in metric_names if name in previous and name in summary]
        pd.DataFrame(comparison).to_csv(output_dir / "comparison_with_residual.csv", index=False, encoding="utf-8-sig")
    output_name = output_dir.name
    current_label = ("reference_modulated" if "reference_modulated" in output_name else
                     "min_snr_uncalibrated" if "min_snr_uncalibrated" in output_name else
                     "variance_selected")
    comparison_sources = {
        "residual": output_dir.parent / "evaluation" / "evaluation_summary.json",
        "absolute_attention": output_dir.parent / "evaluation_absolute_attention" / "evaluation_summary.json",
        "min_snr_uncalibrated": output_dir.parent / "evaluation_absolute_attention_min_snr_uncalibrated" / "evaluation_summary.json",
        current_label: output_dir / "evaluation_summary.json",
    }
    model_rows = []
    for model_name, summary_path in comparison_sources.items():
        if not summary_path.is_file():
            continue
        import json
        values = json.loads(summary_path.read_text(encoding="utf-8"))
        model_rows.append({"model": model_name, **{name: values.get(name) for name in metric_names}})
    if model_rows:
        comparison_frame = pd.DataFrame(model_rows)
        comparison_frame.to_csv(output_dir / "comparison_all_models.csv", index=False, encoding="utf-8-sig")
        if current_label == "reference_modulated":
            comparison_frame.loc[comparison_frame["model"].isin(
                ["min_snr_uncalibrated", "reference_modulated"]
            )].to_csv(output_dir / "comparison_with_min_snr.csv", index=False, encoding="utf-8-sig")
    if plot and not frame.empty:
        selected = frame.sort_values("RMSE", ascending=False).head(int(plot_max_targets))["target_id"].astype(str).tolist()
        samples = _sample_map(prediction_dir / "samples", set(selected))
        history_values, history_masks = np.load(inputs / "history_values.npy"), np.load(inputs / "history_masks.npy")
        guide_values, guide_masks = np.load(inputs / "guide_values.npy"), np.load(inputs / "guide_masks.npy")
        baselines = np.load(inputs / "guide_baselines.npy")
        import json
        absolute_mode = json.loads((dataset_dir / "guide_search_config.json").read_text(encoding="utf-8")).get("target_mode") == "absolute"
        all_actual = _actual_history(dataset_dir)
        meta_lookup = {str(row.target_id): (index, row._asdict()) for index, row in enumerate(metadata.itertuples(index=False))}
        for target_id in progress_bar(selected, enabled=progress, desc="評価画像", unit="image"):
            if target_id not in samples: continue
            data_index, meta = meta_lookup[target_id]; target_index = target_lookup[target_id]
            relevant = all_actual.loc[
                (all_actual["direction"].astype(str) == str(meta["direction"]))
                & (pd.to_numeric(all_actual["bin_start_m"], errors="coerce") == float(meta["bin_start_m"]))
                & pd.to_numeric(all_actual["velocity"], errors="coerce").between(50, 75, inclusive="both")
            ].copy()
            date_label = pd.Timestamp(meta["anchor_date"]).strftime("%Y%m%d")
            filename = f"{float(meta['bin_start_m']):.0f}-{float(meta['bin_end_m']):.0f}m_{meta['direction']}_{date_label}_{target_id}.png"
            plot_target(
                output_dir / "plots" / str(meta["direction"]) / filename, meta,
                history_values[data_index], history_masks[data_index], guide_values[data_index],
                guide_masks[data_index], None if absolute_mode else baselines[data_index],
                predictions.loc[predictions["target_id"].astype(str) == target_id],
                target_values[target_index], target_masks[target_index], samples[target_id], y_max=y_max, dpi=dpi,
                actual_history=relevant, single_sample_index=0,
            )
    return summary
