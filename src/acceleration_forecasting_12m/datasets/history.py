from __future__ import annotations

import numpy as np
import pandas as pd


def select_monthly_history(group, anchor_date, *, months=5, search_days=15):
    anchor = pd.Timestamp(anchor_date).normalize()
    available = group.loc[group["measurement_date"] < anchor].copy()
    anchor_month = anchor.replace(day=1)
    used, values, masks, dates = set(), [], [], []
    for offset in range(months, 0, -1):
        target = anchor_month - pd.DateOffset(months=offset)
        candidates = available.loc[
            (available["measurement_date"] - target).dt.days.abs() <= int(search_days)
        ].copy()
        candidates = candidates.loc[~candidates["measurement_date"].isin(used)]
        if candidates.empty:
            values.append(np.nan); masks.append(0); dates.append("")
            continue
        candidates["_distance"] = (candidates["measurement_date"] - target).dt.days.abs()
        candidates["_future_tie"] = (candidates["measurement_date"] > target).astype(int)
        row = candidates.sort_values(
            ["_distance", "_future_tie", "measurement_date"], kind="mergesort"
        ).iloc[0]
        date = pd.Timestamp(row["measurement_date"]).normalize()
        value = float(row["current_acc_z_max"])
        used.add(date)
        values.append(value if np.isfinite(value) else np.nan)
        masks.append(int(np.isfinite(value)))
        dates.append(date.strftime("%Y-%m-%d") if np.isfinite(value) else "")
    return np.asarray(values, np.float32), np.asarray(masks, np.float32), dates


def truncate_future(row, *, months=12):
    def parse(value):
        import json
        return json.loads(value) if isinstance(value, str) else value
    values, masks, dates = parse(row["future_values"]), parse(row["future_mask"]), parse(row["selected_dates"])
    if len(values) < months or len(masks) < months or len(dates) < months:
        raise ValueError("Future arrays do not contain twelve months")
    values = np.asarray([np.nan if value is None else value for value in values[:months]], np.float32)
    masks = np.asarray(masks[:months], np.float32)
    dates = list(dates[:months])
    masks *= np.isfinite(values)
    cutoff = pd.to_datetime(row.get("cutoff_maintenance_date", ""), errors="coerce")
    if pd.notna(cutoff):
        first = pd.Timestamp(row["measurement_date"]).replace(day=1) + pd.DateOffset(months=1)
        for index in range(months):
            if first + pd.DateOffset(months=index) >= pd.Timestamp(cutoff).normalize():
                masks[index] = 0; values[index] = np.nan; dates[index] = None
    values[masks == 0] = np.nan
    return values, masks, dates

