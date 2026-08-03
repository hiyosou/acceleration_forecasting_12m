from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "MS Gothic"


def plot_target(path, metadata, history_values, history_masks, guide_values, guide_masks,
                baseline, prediction, actual, target_mask, samples, *, y_max=6.0, dpi=150,
                actual_history=None, single_sample_index=0):
    anchor = pd.Timestamp(metadata["anchor_date"])
    history_dates = [pd.Timestamp(value) if value else pd.NaT for value in json.loads(metadata["history_dates"])]
    future_dates = pd.date_range(anchor.replace(day=1) + pd.DateOffset(months=1), periods=12, freq="MS")
    prediction = prediction.sort_values("month_index")
    p10 = prediction["prediction_p10"].to_numpy(float)
    p90 = prediction["prediction_p90"].to_numpy(float)
    sample = np.asarray(samples[int(single_sample_index)], dtype=float)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for axis in (top, bottom): axis.set_facecolor("none")

    if actual_history is not None and not actual_history.empty:
        speed = pd.to_numeric(actual_history.get("velocity"), errors="coerce")
        finite_speed = speed.notna()
        speed_scatter = None
        if finite_speed.any():
            speed_scatter = top.scatter(
                pd.to_datetime(actual_history.loc[finite_speed, "measurement_date"]),
                actual_history.loc[finite_speed, "current_acc_z_max"],
                s=24, c=speed.loc[finite_speed], cmap="jet", vmin=0, vmax=100,
                alpha=0.72, label="全実測値", zorder=2,
            )
        if (~finite_speed).any():
            top.scatter(
                pd.to_datetime(actual_history.loc[~finite_speed, "measurement_date"]),
                actual_history.loc[~finite_speed, "current_acc_z_max"],
                s=24, color="0.55", alpha=0.65, label="全実測値（速度欠損）", zorder=2,
            )
        if speed_scatter is not None:
            colorbar = fig.colorbar(speed_scatter, ax=top, pad=0.01)
            colorbar.set_label("走行速度 [km/h]")
    valid_history = np.asarray(history_masks, bool)
    history_date_array = np.asarray(history_dates, dtype=object)
    top.plot(history_date_array[valid_history], np.asarray(history_values)[valid_history],
             "s-", color="darkorange", linewidth=1.5, label="モデル入力：過去5か月")
    top.scatter([anchor], [float(metadata["current_acc_z_max"])], marker="*", s=130,
                color="steelblue", zorder=8, label="現在値・予測起点")
    colors = ("tab:blue", "tab:green", "tab:purple")
    for rank in range(3):
        valid = np.asarray(guide_masks[rank], bool)
        if valid.any():
            top.plot(future_dates, np.where(valid, guide_values[rank], np.nan), "--",
                     color=colors[rank], alpha=0.8, label=f"Guide {rank + 1}")
    if baseline is not None:
        top.plot(future_dates, baseline, color="darkorange", linewidth=2, label="Softmax guide baseline")
    top.fill_between(future_dates, p10, p90, color="red", alpha=0.15, label="予測p10-p90")
    top.plot(future_dates, sample, "o-", color="red", linewidth=2,
             label=f"生成例 sample {single_sample_index}")
    valid_target = np.asarray(target_mask, bool)
    top.plot(future_dates[valid_target], np.asarray(actual)[valid_target], "o-", color="black",
             markerfacecolor="white", label="未来正解")
    top.axvline(anchor, color="steelblue", linewidth=1.5)
    top.axvline(future_dates[0], color="navy", linestyle="--", linewidth=2, label="予測開始")
    cutoff = pd.to_datetime(metadata.get("cutoff_maintenance_date", ""), errors="coerce")
    if pd.notna(cutoff): top.axvline(cutoff, color="0.35", linestyle=":", label="施工日")
    top.set(title=f"{metadata['direction']}方向 {metadata['bin_start_m']:.0f}-{metadata['bin_end_m']:.0f}m / 起点 {metadata['anchor_date']}",
            ylabel="絶対値最大加速度 [m/s²]", ylim=(0, y_max))
    top.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); top.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    top.legend(fontsize=8, ncol=2); top.grid(True, color="0.85")

    months = np.arange(1, 13)
    for generated in samples: bottom.plot(months, generated, color="red", alpha=0.04, linewidth=0.7)
    bottom.fill_between(months, p10, p90, color="red", alpha=0.15)
    bottom.plot(months, sample, color="darkred", linewidth=2.2, label=f"生成例 sample {single_sample_index}")
    if baseline is not None: bottom.plot(months, baseline, color="darkorange", linewidth=2, label="guide baseline")
    bottom.plot(months[valid_target], np.asarray(actual)[valid_target], "o-", color="black",
                markerfacecolor="white", label="未来正解")
    bottom.set(xlabel="予測先 [か月]", ylabel="絶対値最大加速度 [m/s²]", xlim=(1, 12), ylim=(0, y_max))
    bottom.set_xticks(months); bottom.grid(True, color="0.85"); bottom.legend()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, transparent=True); plt.close(fig)
