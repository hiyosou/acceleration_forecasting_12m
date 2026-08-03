from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from acceleration_forecasting_12m.retrieval.autoencoder import WaveformAutoencoder


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    dimension = int(checkpoint.get("embedding_dim", 256))
    model = WaveformAutoencoder(dimension).to(device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    return model, checkpoint


def _metric_summary(path):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return frame, {name: float(frame[name].mean(skipna=True)) for name in (
        "mae", "rmse", "correlation", "peak_amplitude_error", "peak_distance_error_m"
    )}


def _benchmark_encoding(model, checkpoint, source_dir, device, records=8192, batch_size=512):
    source = _read_json(Path(source_dir) / "source_artifacts.json")
    count = min(int(records), int(source["record_count"]))
    waveforms = np.memmap(source["waveforms_path"], dtype=np.float32, mode="r",
                          shape=(int(source["record_count"]), 500))
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            raw = np.asarray(waveforms[start:min(start + batch_size, count)], np.float32).copy()
            tensor = torch.from_numpy((raw - float(checkpoint["mean"])) / float(checkpoint["std"]))[:, None].to(device)
            model.encode(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return elapsed, count / max(elapsed, 1e-12)


def _guide_quality(dataset_dir):
    dataset_dir = Path(dataset_dir)
    values = np.load(dataset_dir / "inference" / "inputs" / "guide_values.npy", mmap_mode="r")
    masks = np.load(dataset_dir / "inference" / "inputs" / "guide_masks.npy", mmap_mode="r")
    targets = np.load(dataset_dir / "inference" / "targets" / "target_values.npy", mmap_mode="r")
    target_masks = np.load(dataset_dir / "inference" / "targets" / "target_masks.npy", mmap_mode="r")
    errors, correlations = [], []
    for guide, guide_mask, target, target_mask in zip(values, masks, targets, target_masks):
        for series, series_mask in zip(guide, guide_mask):
            valid = (series_mask > 0) & (target_mask > 0) & np.isfinite(series) & np.isfinite(target)
            if not valid.any():
                continue
            errors.append(float(np.mean(np.abs(series[valid] - target[valid]))))
            if valid.sum() >= 2 and np.std(series[valid]) > 0 and np.std(target[valid]) > 0:
                correlations.append(float(np.corrcoef(series[valid], target[valid])[0, 1]))
    return float(np.mean(errors)), float(np.mean(correlations)) if correlations else np.nan


def _search_summary(assignments_path, baseline_path=None):
    frame = pd.read_csv(assignments_path, encoding="utf-8-sig")
    selected = frame.loc[frame["selection_status"] == "selected"].copy()
    counts = selected.groupby(["split", "target_id"]).size()
    result = {
        "selected_guides": int(len(selected)),
        "mean_cosine_similarity": float(selected["cosine_similarity"].mean()),
        "median_cosine_similarity": float(selected["cosine_similarity"].median()),
        "three_guide_rate": float((counts == 3).mean()),
        "insufficient_guide_targets": int((counts < 3).sum()),
    }
    if baseline_path is not None:
        baseline = pd.read_csv(baseline_path, encoding="utf-8-sig")
        baseline = baseline.loc[baseline["selection_status"] == "selected"]
        top_current = selected.sort_values("guide_rank").groupby(["split", "target_id"]).first()
        top_base = baseline.sort_values("guide_rank").groupby(["split", "target_id"]).first()
        common = top_current.index.intersection(top_base.index)
        result["top1_match_rate_vs_256"] = float((
            top_current.loc[common, "guide_trend_id"].astype(str).to_numpy()
            == top_base.loc[common, "guide_trend_id"].astype(str).to_numpy()
        ).mean())
        current_sets = selected.groupby(["split", "target_id"])["guide_trend_id"].agg(lambda x: set(x.astype(str)))
        base_sets = baseline.groupby(["split", "target_id"])["guide_trend_id"].agg(lambda x: set(x.astype(str)))
        common = current_sets.index.intersection(base_sets.index)
        overlaps = [len(current_sets.loc[key] & base_sets.loc[key]) / 3.0 for key in common]
        exact = [current_sets.loc[key] == base_sets.loc[key] for key in common]
        result["mean_top3_overlap_vs_256"] = float(np.mean(overlaps))
        result["exact_top3_match_rate_vs_256"] = float(np.mean(exact))
    return result


def _plot_representative(root, metric_frames, checkpoints, source_artifact_dir, output_path, device):
    baseline = metric_frames[256].sort_values("rmse").reset_index(drop=True)
    row = baseline.iloc[len(baseline) // 2]
    record_id, waveform_index = str(row["record_id"]), int(row["waveform_index"])
    source = _read_json(Path(source_artifact_dir) / "source_artifacts.json")
    waveform_path = Path(source["waveforms_path"])
    record_count = int(source["record_count"])
    waveforms = np.memmap(waveform_path, dtype=np.float32, mode="r", shape=(record_count, 500))
    original = np.asarray(waveforms[waveform_index], np.float32).copy()
    reconstructions, rmses = {}, {}
    for dimension, checkpoint_path in checkpoints.items():
        model, checkpoint = _load_model(checkpoint_path, device)
        normalized = torch.from_numpy(((original - float(checkpoint["mean"])) / float(checkpoint["std"])).copy())[None, None].to(device)
        with torch.inference_mode():
            reconstructed, _ = model(normalized)
        physical = reconstructed[0, 0].cpu().numpy() * float(checkpoint["std"]) + float(checkpoint["mean"])
        reconstructions[dimension] = physical
        rmses[dimension] = float(np.sqrt(np.mean((physical - original) ** 2)))
    distance = float(row["bin_start_m"]) + np.arange(500) * 0.2
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(distance, original, color="black", lw=1.3, label="Original")
    colors = {256: "red", 64: "royalblue", 32: "darkorange"}
    for dimension in (256, 64, 32):
        axis.plot(distance, reconstructions[dimension], color=colors[dimension], lw=1.0,
                  label=f"{dimension}D reconstruction (RMSE={rmses[dimension]:.4f})")
    axis.set_xlabel("Corrected distance [m]"); axis.set_ylabel("acc_z [m/s²]")
    axis.set_title(f"{row['measurement_date']} / {row['direction']} / {row['bin_start_m']:.0f}-{row['bin_end_m']:.0f}m / {record_id}")
    axis.grid(color="0.85"); axis.legend(); fig.tight_layout()
    fig.savefig(output_path, dpi=180, transparent=True); plt.close(fig)
    return {"record_id": record_id, "waveform_index": waveform_index, "rmse": rmses}


def build_comparison(root_dir, source_256, *, device=None):
    root, source_256 = Path(root_dir).resolve(), Path(source_256).resolve()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dimensions = {
        256: {
            "source": source_256,
            "metrics": root / "dim_256_autoencoder_eval" / "trained" / "reconstruction_metrics.csv",
            "datasets": Path(__file__).resolve().parents[3] / "artifacts" / "datasets_absolute_attention",
            "evaluation": Path(__file__).resolve().parents[3] / "artifacts" / "evaluation_absolute_reference_modulated_v",
        },
        64: {"source": root / "dim_64" / "source", "metrics": root / "dim_64" / "autoencoder_eval" / "trained" / "reconstruction_metrics.csv", "datasets": root / "dim_64" / "datasets", "evaluation": root / "dim_64" / "evaluation"},
        32: {"source": root / "dim_32" / "source", "metrics": root / "dim_32" / "autoencoder_eval" / "trained" / "reconstruction_metrics.csv", "datasets": root / "dim_32" / "datasets", "evaluation": root / "dim_32" / "evaluation"},
    }
    metric_frames, ae_rows, forecast_rows, search_rows, checkpoints = {}, [], [], [], {}
    baseline_assignments = dimensions[256]["datasets"] / "guide_assignments.csv"
    for dimension, paths in dimensions.items():
        frame, metrics = _metric_summary(paths["metrics"]); metric_frames[dimension] = frame
        checkpoint = paths["source"] / "autoencoder.pt"; checkpoints[dimension] = checkpoint
        history = pd.read_csv(paths["source"] / "training_history.csv", encoding="utf-8-sig")
        model, checkpoint_data = _load_model(checkpoint, device)
        encode_seconds, records_per_second = _benchmark_encoding(model, checkpoint_data, paths["source"], device)
        ae_rows.append({"embedding_dim": dimension, "best_valid_smooth_l1": float(history["valid_loss"].min()),
                        **metrics, "parameter_count": sum(p.numel() for p in model.parameters()),
                        "checkpoint_bytes": checkpoint.stat().st_size,
                        "database_bytes": (paths["source"] / "vector_database.sqlite").stat().st_size,
                        "encoding_benchmark_records": 8192, "encoding_seconds": encode_seconds,
                        "encoding_records_per_second": records_per_second})
        search = _search_summary(paths["datasets"] / "guide_assignments.csv", None if dimension == 256 else baseline_assignments)
        guide_mae, guide_corr = _guide_quality(paths["datasets"]); search.update({"embedding_dim": dimension, "guide_future_mae": guide_mae, "guide_future_correlation": guide_corr})
        search_rows.append(search)
        summary = _read_json(paths["evaluation"] / "evaluation_summary.json")
        forecast_rows.append({"embedding_dim": dimension, **summary})
    pd.DataFrame(ae_rows).to_csv(root / "autoencoder_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(search_rows).to_csv(root / "retrieval_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(forecast_rows).to_csv(root / "forecasting_comparison.csv", index=False, encoding="utf-8-sig")
    representative = _plot_representative(root, metric_frames, checkpoints, source_256,
                                            root / "representative_reconstruction_256_64_32.png", device)
    result = {"dimensions": [256, 64, 32], "representative": representative,
              "autoencoder_comparison": str(root / "autoencoder_comparison.csv"),
              "retrieval_comparison": str(root / "retrieval_comparison.csv"),
              "forecasting_comparison": str(root / "forecasting_comparison.csv")}
    (root / "comparison_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
