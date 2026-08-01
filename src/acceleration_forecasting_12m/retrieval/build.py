from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from acceleration_forecasting_12m.common.constants import EMBEDDING_DIM, FORECAST_MONTHS, MIN_GUIDE_MONTHS
from acceleration_forecasting_12m.common.io import artifact_descriptor, write_json
from acceleration_forecasting_12m.common.progress import progress_bar
from .database import SCHEMA, connect_read_only


SOURCE_TREND_COLUMNS = (
    "trend_id", "dataset_id", "measurement_date", "direction", "bin_start_m", "bin_end_m",
    "current_acc_z_max", "future_values", "future_mask", "selected_dates",
    "cutoff_maintenance_date", "maintenance_type", "maintenance_description",
)
WAVEFORM_COLUMNS = (
    "record_id", "measurement_id", "measurement_date", "direction", "bin_start_m", "bin_end_m",
    "mean_velocity_kmh", "source_csv_path", "waveform_sha256", "dataset_id", "trend_id",
    "embedding", "embedding_dim",
)


def _as_list(value):
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list")
    return parsed


def _truncate_future(row, months, min_valid):
    try:
        current = float(row[6])
        values = _as_list(row[7])
        masks = _as_list(row[8])
        dates = _as_list(row[9])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, "invalid_future_shape", 0
    if not np.isfinite(current):
        return None, "non_finite_current", 0
    if len(values) < months or len(masks) < months or len(dates) < months:
        return None, "invalid_future_shape", 0
    values, masks, dates = values[:months], masks[:months], dates[:months]
    output_values, output_masks, output_dates = [], [], []
    valid = 0
    for value, mask, selected_date in zip(values, masks, dates):
        if bool(mask):
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None, "non_finite_future", valid
            if not np.isfinite(number):
                return None, "non_finite_future", valid
            output_values.append(number)
            output_masks.append(1)
            output_dates.append(selected_date or None)
            valid += 1
        else:
            output_values.append(None)
            output_masks.append(0)
            output_dates.append(None)
    if valid < min_valid:
        return None, "insufficient_valid_months", valid
    return (current, output_values, output_masks, output_dates), "eligible", valid


def _embedding_status(blob, dimension, expected):
    try:
        dimension = int(dimension)
    except (TypeError, ValueError):
        return "invalid_embedding"
    if dimension != expected or len(blob) != expected * 4:
        return "invalid_embedding"
    vector = np.frombuffer(blob, dtype=np.float32, count=expected)
    if not np.isfinite(vector).all():
        return "invalid_embedding"
    if float(np.linalg.norm(vector)) <= 1e-12:
        return "zero_norm_embedding"
    return "valid"


def build_retrieval_database(source_artifact_dir, output_dir, *, months=FORECAST_MONTHS,
                             min_valid_months=MIN_GUIDE_MONTHS,
                             embedding_dim=EMBEDDING_DIM, progress=True):
    source_dir, output_dir = Path(source_artifact_dir).resolve(), Path(output_dir).resolve()
    source_database = source_dir / "vector_database.sqlite"
    manifest_path = source_dir / "dataset_split_manifest.csv"
    if not source_database.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Source database or split manifest is missing")
    output_dir.mkdir(parents=True, exist_ok=True)
    final_db = output_dir / "vector_database_12m.sqlite"
    temp_db = output_dir / ".vector_database_12m.sqlite.tmp"
    if temp_db.exists():
        temp_db.unlink()

    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", usecols=["dataset_id", "model_split"])
    split_map = manifest.drop_duplicates("dataset_id").set_index("dataset_id")["model_split"].astype(str).to_dict()
    source, target = connect_read_only(source_database), sqlite3.connect(temp_db)
    target.executescript(SCHEMA)
    diagnostics, trend_rows = {}, {}
    try:
        trend_count = int(source.execute("SELECT COUNT(*) FROM trends").fetchone()[0])
        query = f"SELECT {','.join(SOURCE_TREND_COLUMNS)} FROM trends ORDER BY trend_id"
        for row in progress_bar(source.execute(query), enabled=progress, total=trend_count,
                                desc="12か月ガイドを検査", unit="trend"):
            parsed, status, valid = _truncate_future(row, months, min_valid_months)
            trend_id, dataset_id = str(row[0]), str(row[1])
            item = {
                "trend_id": trend_id, "dataset_id": dataset_id, "measurement_date": str(row[2]),
                "valid_future_months": int(valid), "source_waveform_count": 0,
                "valid_waveform_count": 0, "excluded_waveform_count": 0, "status": status,
            }
            diagnostics[trend_id] = item
            if parsed is None:
                continue
            model_split = split_map.get(dataset_id, "")
            if model_split not in {"model_train", "model_validation"}:
                item["status"] = "invalid_split"
                continue
            current, values, masks, selected_dates = parsed
            anchor_month = pd.Timestamp(row[2]).replace(day=1)
            available_date = anchor_month + pd.DateOffset(months=months) + pd.Timedelta(days=15)
            trend_rows[trend_id] = (
                trend_id, dataset_id, model_split, str(row[2]), str(row[3]), float(row[4]), float(row[5]),
                current, json.dumps(values), json.dumps(masks), json.dumps(selected_dates),
                available_date.strftime("%Y-%m-%d"), row[10], row[11], row[12],
            )

        waveform_count = int(source.execute("SELECT COUNT(*) FROM waveform_records").fetchone()[0])
        query = f"SELECT {','.join(WAVEFORM_COLUMNS)} FROM waveform_records ORDER BY record_id"
        valid_waveforms = defaultdict(list)
        waveform_reason_counts = Counter()
        for row in progress_bar(source.execute(query), enabled=progress, total=waveform_count,
                                desc="256次元波形を検査", unit="waveform"):
            trend_id = str(row[10])
            if trend_id not in trend_rows:
                continue
            diagnostics[trend_id]["source_waveform_count"] += 1
            status = _embedding_status(row[11], row[12], embedding_dim)
            if status == "valid":
                valid_waveforms[trend_id].append(row)
                diagnostics[trend_id]["valid_waveform_count"] += 1
            else:
                diagnostics[trend_id]["excluded_waveform_count"] += 1
                waveform_reason_counts[status] += 1

        adopted_trends, adopted_waveforms = [], []
        for trend_id, row in trend_rows.items():
            waveforms = valid_waveforms.get(trend_id, [])
            if not waveforms:
                diagnostics[trend_id]["status"] = (
                    "missing_waveform" if diagnostics[trend_id]["source_waveform_count"] == 0
                    else "invalid_embedding"
                )
                continue
            diagnostics[trend_id]["status"] = "adopted"
            adopted_trends.append(row)
            adopted_waveforms.extend(waveforms)

        trend_columns = (
            "trend_id", "dataset_id", "model_split", "measurement_date", "direction", "bin_start_m",
            "bin_end_m", "current_acc_z_max", "future_values", "future_mask", "selected_dates",
            "guide_available_date", "cutoff_maintenance_date", "maintenance_type", "maintenance_description",
        )
        target.executemany(
            f"INSERT INTO trends({','.join(trend_columns)}) VALUES ({','.join('?' for _ in trend_columns)})",
            adopted_trends,
        )
        target.executemany(
            f"INSERT INTO waveform_records({','.join(WAVEFORM_COLUMNS)}) VALUES ({','.join('?' for _ in WAVEFORM_COLUMNS)})",
            adopted_waveforms,
        )
        descriptors = {
            "source_database": artifact_descriptor(source_database),
            "source_manifest": artifact_descriptor(manifest_path),
        }
        metadata = {
            "forecast_months": months, "min_valid_months": min_valid_months,
            "embedding_dim": embedding_dim, "source_artifacts": descriptors,
        }
        target.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)",
            [(key, json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()],
        )
        target.commit()
        if target.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Foreign-key validation failed")
        if target.execute("SELECT COUNT(*) FROM waveform_records w LEFT JOIN trends t ON t.trend_id=w.trend_id WHERE t.trend_id IS NULL").fetchone()[0]:
            raise RuntimeError("Orphan waveform detected")
        diagnostic_frame = pd.DataFrame(diagnostics.values()).sort_values("trend_id")
        status_counts = diagnostic_frame["status"].value_counts().to_dict()
        summary = {
            "source_artifact_dir": str(source_dir), "forecast_months": int(months),
            "min_valid_months": int(min_valid_months), "embedding_dim": int(embedding_dim),
            "source_trend_count": trend_count, "source_waveform_count": waveform_count,
            "adopted_trend_count": len(adopted_trends), "adopted_waveform_count": len(adopted_waveforms),
            "adopted_dataset_count": int(target.execute("SELECT COUNT(DISTINCT dataset_id) FROM trends").fetchone()[0]),
            "status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "invalid_waveform_counts": {str(k): int(v) for k, v in waveform_reason_counts.items()},
            "database_path": str(final_db), "source_artifacts": descriptors,
        }
        diagnostic_frame.to_csv(output_dir / "exclusion_diagnostics.csv", index=False, encoding="utf-8-sig")
    except Exception:
        target.close(); source.close()
        if temp_db.exists():
            temp_db.unlink()
        raise
    target.close(); source.close()
    os.replace(temp_db, final_db)
    write_json(output_dir / "database_summary.json", summary)
    return summary

