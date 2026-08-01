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
                baseline, prediction, actual, target_mask, samples, *, y_max=6.0, dpi=150):
    anchor = pd.Timestamp(metadata["anchor_date"])
    history_dates = [pd.Timestamp(value) if value else pd.NaT for value in json.loads(metadata["history_dates"])]
    future_dates = pd.date_range(anchor.replace(day=1) + pd.DateOffset(months=1), periods=12, freq="MS")
    prediction = prediction.sort_values("month_index")
    median = prediction["prediction_median"].to_numpy(float)
    p10, p90 = prediction["prediction_p10"].to_numpy(float), prediction["prediction_p90"].to_numpy(float)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for axis in (top, bottom): axis.set_facecolor("none")
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
    top.plot(future_dates, baseline, color="darkorange", linewidth=2, label="Softmaxガイド基準")
    top.fill_between(future_dates, p10, p90, color="red", alpha=0.15, label="予測p10-p90")
    top.plot(future_dates, median, "o-", color="red", linewidth=2, label="予測中央値")
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
    for sample in samples: bottom.plot(months, sample, color="red", alpha=0.04, linewidth=0.7)
    bottom.fill_between(months, p10, p90, color="red", alpha=0.15)
    bottom.plot(months, median, color="red", linewidth=2, label="予測中央値")
    bottom.plot(months, baseline, color="darkorange", linewidth=2, label="ガイド基準")
    bottom.plot(months[valid_target], np.asarray(actual)[valid_target], "o-", color="black",
                markerfacecolor="white", label="未来正解")
    bottom.set(xlabel="予測先 [か月]", ylabel="絶対値最大加速度 [m/s²]", xlim=(1, 12), ylim=(0, y_max))
    bottom.set_xticks(months); bottom.grid(True, color="0.85"); bottom.legend()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, transparent=True); plt.close(fig)
