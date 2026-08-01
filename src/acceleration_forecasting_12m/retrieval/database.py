from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE trends(
    trend_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    model_split TEXT NOT NULL,
    measurement_date TEXT NOT NULL,
    direction TEXT NOT NULL,
    bin_start_m REAL NOT NULL,
    bin_end_m REAL NOT NULL,
    current_acc_z_max REAL NOT NULL,
    future_values TEXT NOT NULL,
    future_mask TEXT NOT NULL,
    selected_dates TEXT NOT NULL,
    guide_available_date TEXT NOT NULL,
    cutoff_maintenance_date TEXT,
    maintenance_type TEXT,
    maintenance_description TEXT
);
CREATE TABLE waveform_records(
    record_id TEXT PRIMARY KEY,
    measurement_id TEXT NOT NULL,
    measurement_date TEXT NOT NULL,
    direction TEXT NOT NULL,
    bin_start_m REAL NOT NULL,
    bin_end_m REAL NOT NULL,
    mean_velocity_kmh REAL NOT NULL,
    source_csv_path TEXT NOT NULL,
    waveform_sha256 TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    trend_id TEXT NOT NULL REFERENCES trends(trend_id),
    embedding BLOB NOT NULL,
    embedding_dim INTEGER NOT NULL
);
CREATE INDEX idx_trends_split ON trends(model_split);
CREATE INDEX idx_trends_dataset ON trends(dataset_id);
CREATE INDEX idx_waveform_trend ON waveform_records(trend_id);
CREATE INDEX idx_waveform_date ON waveform_records(measurement_date);
"""


def connect_read_only(path):
    path = Path(path).resolve()
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)

