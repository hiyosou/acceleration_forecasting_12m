from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from acceleration_forecasting_12m.common.io import write_json
from acceleration_forecasting_12m.common.progress import progress_bar
from acceleration_forecasting_12m.common.constants import PHYSICAL_MAX, PHYSICAL_MIN
from acceleration_forecasting_12m.datasets.torch_dataset import ForecastDataset
from .sampling import load_process, sample_target


def predict(dataset_dir, checkpoint, output_dir, *, device=None, num_samples=100,
            sampling_steps=50, save_samples=True, max_records=None, seed=42, progress=True):
    dataset_dir, output_dir = Path(dataset_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True); sample_dir = output_dir / "samples"; sample_dir.mkdir(exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    process, checkpoint_data = load_process(checkpoint, device)
    dataset = ForecastDataset(dataset_dir / "inference" / "inputs", dataset_dir, include_targets=False)
    count = min(len(dataset), int(max_records)) if max_records is not None else len(dataset)
    requested_ids = dataset.metadata.iloc[:count]["target_id"].astype(str).tolist()
    existing_rows = pd.DataFrame()
    predictions_path = output_dir / "predictions.csv"
    run_path = output_dir / "prediction_run.json"
    if predictions_path.is_file() and run_path.is_file():
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        compatible = (
            previous.get("dataset_build_id") == checkpoint_data["dataset_build_id"]
            and int(previous.get("num_samples", -1)) == int(num_samples)
            and int(previous.get("sampling_steps", -1)) == int(sampling_steps)
            and str(previous.get("checkpoint")) == str(Path(checkpoint).resolve())
            and previous.get("physical_bounds") == [PHYSICAL_MIN, PHYSICAL_MAX]
        )
        if not compatible:
            raise ValueError("Existing prediction output was created with incompatible settings")
        existing_rows = pd.read_csv(predictions_path, encoding="utf-8-sig")
    completed_ids = set()
    if not existing_rows.empty:
        counts = existing_rows.groupby(existing_rows["target_id"].astype(str)).size()
        completed_ids = {target_id for target_id, size in counts.items() if int(size) == 12}
    pending_indices = [index for index, target_id in enumerate(requested_ids) if target_id not in completed_ids]
    rows, chunk_ids, chunk_samples = [], [], []
    started = time.perf_counter()
    for index in progress_bar(pending_indices, enabled=progress, total=len(pending_indices), desc="12か月推論", unit="target"):
        target_id = str(dataset.metadata.iloc[index]["target_id"])
        samples = sample_target(process, dataset, index, target_id, num_samples=num_samples,
                                sampling_steps=sampling_steps, device=device, seed=seed)
        summary = {
            "mean": samples.mean(axis=0), "median": np.median(samples, axis=0),
            "p10": np.percentile(samples, 10, axis=0), "p90": np.percentile(samples, 90, axis=0),
            "std": samples.std(axis=0),
        }
        for month in range(12):
            rows.append({
                "target_id": target_id, "month_index": month + 1,
                **{f"prediction_{name}": float(values[month]) for name, values in summary.items()},
            })
        if save_samples:
            chunk_ids.append(target_id); chunk_samples.append(samples.astype(np.float32))
            if len(chunk_ids) >= 32:
                chunk_number = len(list(sample_dir.glob("samples_*.npz")))
                np.savez_compressed(sample_dir / f"samples_{chunk_number:04d}.npz",
                                    target_ids=np.asarray(chunk_ids), samples=np.asarray(chunk_samples))
                chunk_ids, chunk_samples = [], []
    if save_samples and chunk_ids:
        chunk_number = len(list(sample_dir.glob("samples_*.npz")))
        np.savez_compressed(sample_dir / f"samples_{chunk_number:04d}.npz",
                            target_ids=np.asarray(chunk_ids), samples=np.asarray(chunk_samples))
    new_rows = pd.DataFrame(rows)
    combined = pd.concat([existing_rows, new_rows], ignore_index=True)
    if not combined.empty:
        combined = combined.loc[combined["target_id"].astype(str).isin(requested_ids)]
        order = {target_id: index for index, target_id in enumerate(requested_ids)}
        combined["_target_order"] = combined["target_id"].astype(str).map(order)
        combined = combined.sort_values(["_target_order", "month_index"]).drop(columns="_target_order")
    combined.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    guide_source = dataset_dir / "guide_assignments.csv"
    if guide_source.is_file():
        guides = pd.read_csv(guide_source, encoding="utf-8-sig")
        wanted = set(dataset.metadata.iloc[:count]["target_id"].astype(str))
        guides.loc[guides["target_id"].astype(str).isin(wanted)].to_csv(
            output_dir / "prediction_guides.csv", index=False, encoding="utf-8-sig"
        )
    result = {
        "target_count": count, "completed_before_run": len(completed_ids & set(requested_ids)),
        "generated_this_run": len(pending_indices),
        "num_samples": int(num_samples), "sampling_steps": int(sampling_steps),
        "physical_bounds": [PHYSICAL_MIN, PHYSICAL_MAX],
        "elapsed_seconds": time.perf_counter() - started, "device": str(device),
        "checkpoint": str(Path(checkpoint).resolve()), "dataset_build_id": checkpoint_data["dataset_build_id"],
    }
    write_json(run_path, result); return result
