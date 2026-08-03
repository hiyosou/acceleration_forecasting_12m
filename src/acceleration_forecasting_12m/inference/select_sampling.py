from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from acceleration_forecasting_12m.common.io import write_json
from acceleration_forecasting_12m.common.progress import progress_bar
from .validate import validate


def select_sampling(dataset_dir, checkpoint, output_dir, *, device=None, num_samples=100,
                    max_records=None, seed=42, progress=True):
    output_dir = Path(output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    combinations = [(scale, steps) for scale in (1.0, 0.75, 0.5) for steps in (50, 100, 200)]
    for scale, steps in progress_bar(combinations, enabled=progress, desc="DDIM設定比較", unit="config"):
        started = time.perf_counter()
        result = validate(
            dataset_dir, checkpoint, output_dir / f"scale_{scale:g}_steps_{steps}",
            device=device, num_samples=num_samples, sampling_steps=steps,
            max_records=max_records, seed=seed, progress=False, initial_noise_scale=scale,
        )
        rows.append({**result, "elapsed_seconds": time.perf_counter() - started})
    frame = pd.DataFrame(rows)
    frame["finite_and_ordered"] = frame["all_finite"].astype(bool)
    frame["coverage_pass"] = frame["coverage_p10_p90"] >= 0.75
    frame["mae_pass"] = frame["MAE"] <= 0.4009
    frame["width_0_7_pass"] = frame["mean_interval_width"] <= 0.7
    frame["width_0_5_pass"] = frame["mean_interval_width"] <= 0.5
    eligible = frame.loc[frame["finite_and_ordered"] & frame["coverage_pass"] & frame["mae_pass"]]
    if not eligible.empty:
        selected = eligible.sort_values(["mean_interval_width", "sampling_steps"], kind="mergesort").iloc[0]
        reason = "coverage>=75%, MAE<=0.4009; minimum interval width"
    else:
        fallback = frame.loc[frame["finite_and_ordered"] & frame["coverage_pass"]]
        if fallback.empty: fallback = frame.loc[frame["finite_and_ordered"]]
        selected = fallback.sort_values(["MAE", "mean_interval_width", "sampling_steps"], kind="mergesort").iloc[0]
        reason = "quality targets unmet; minimum MAE among valid fallback configurations"
    frame.to_csv(output_dir / "sampling_grid.csv", index=False, encoding="utf-8-sig")
    result = {
        "initial_noise_scale": float(selected["initial_noise_scale"]),
        "sampling_steps": int(selected["sampling_steps"]), "num_samples": int(num_samples),
        "selection_reason": reason, "metrics": {
            key: float(selected[key]) for key in ("MAE", "RMSE", "coverage_p10_p90",
                                                   "mean_interval_width", "ensemble_mean_std")
        },
        "width_0_7_achieved": bool(selected["width_0_7_pass"]),
        "width_0_5_achieved": bool(selected["width_0_5_pass"]),
    }
    write_json(output_dir / "selected_sampling_config.json", result)
    return result
