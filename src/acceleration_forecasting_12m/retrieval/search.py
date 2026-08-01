from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SearchConfig:
    top_k: int = 3
    max_current_difference: float = 0.5
    min_valid_months: int = 8
    near_distance_m: float = 100.0
    spatial_tolerance_m: float = 1e-6


class GuideIndex:
    def __init__(self, connection: sqlite3.Connection):
        rows = connection.execute("""
            SELECT w.record_id,w.measurement_id,w.measurement_date,w.embedding,w.embedding_dim,
                   t.trend_id,t.dataset_id,t.model_split,t.current_acc_z_max,t.future_values,
                   t.future_mask,t.selected_dates,t.guide_available_date,t.direction,
                   t.bin_start_m,t.bin_end_m,t.cutoff_maintenance_date,
                   t.maintenance_type,t.maintenance_description
            FROM waveform_records w JOIN trends t ON t.trend_id=w.trend_id
        """).fetchall()
        if not rows:
            raise ValueError("Guide database is empty")
        dimension = int(rows[0][4])
        self.embeddings = np.stack([np.frombuffer(row[3], dtype=np.float32, count=dimension) for row in rows])
        self.record_ids = np.asarray([row[0] for row in rows], dtype=object)
        self.measurement_ids = np.asarray([row[1] for row in rows], dtype=object)
        self.dates = np.asarray([str(row[2]) for row in rows], dtype="U10")
        self.trend_ids = np.asarray([row[5] for row in rows], dtype=object)
        self.datasets = np.asarray([row[6] for row in rows], dtype=object)
        self.splits = np.asarray([row[7] for row in rows], dtype=object)
        self.current = np.asarray([row[8] for row in rows], dtype=np.float32)
        self.future_values = [json.loads(row[9]) for row in rows]
        self.future_masks = [json.loads(row[10]) for row in rows]
        self.selected_dates = [json.loads(row[11]) for row in rows]
        self.available_dates = np.asarray([str(row[12]) for row in rows], dtype="U10")
        self.directions = np.asarray([row[13] for row in rows], dtype=object)
        self.bin_starts = np.asarray([row[14] for row in rows], dtype=np.float32)
        self.bin_ends = np.asarray([row[15] for row in rows], dtype=np.float32)
        self.cutoffs = [row[16] or "" for row in rows]
        self.maintenance_types = [row[17] or "" for row in rows]
        self.maintenance_descriptions = [row[18] or "" for row in rows]
        self.valid_months = np.asarray([sum(bool(value) for value in mask) for mask in self.future_masks], dtype=np.int16)

    def search(self, query_embeddings, *, query_date, query_dataset_id, query_current,
               query_bin_start_m, allowed_splits, config=SearchConfig()):
        query = np.asarray(query_embeddings, dtype=np.float32)
        if query.ndim == 1:
            query = query[None, :]
        norms = np.linalg.norm(query, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or (norms <= 1e-12).any():
            raise ValueError("Query embeddings must be finite and non-zero")
        query = query / norms
        allowed_splits = set(str(value) for value in allowed_splits)
        eligible = np.fromiter((str(value) in allowed_splits for value in self.splits), dtype=bool)
        eligible &= self.datasets != str(query_dataset_id)
        eligible &= self.dates != str(query_date)
        eligible &= np.isfinite(self.current)
        eligible &= np.abs(self.current - float(query_current)) <= config.max_current_difference + 1e-12
        eligible &= self.valid_months >= config.min_valid_months
        distance = np.abs(self.bin_starts.astype(np.float64) - float(query_bin_start_m))
        near = distance <= config.near_distance_m + config.spatial_tolerance_m
        temporal_applied = near
        eligible &= ~near | (self.available_dates < str(query_date))
        indices = np.flatnonzero(eligible)
        if not len(indices):
            return []
        similarities = query @ self.embeddings[indices].T
        best = similarities.max(axis=0)
        order = np.argsort(-best, kind="stable")
        selected, used_dates = [], set()
        for offset in order:
            index = int(indices[int(offset)])
            date = str(self.dates[index])
            if date in used_dates:
                continue
            used_dates.add(date)
            selected.append({
                "record_id": self.record_ids[index], "measurement_id": self.measurement_ids[index],
                "measurement_date": date, "trend_id": self.trend_ids[index],
                "dataset_id": self.datasets[index], "model_split": self.splits[index],
                "current_acc_z_max": float(self.current[index]),
                "current_max_difference": float(self.current[index] - query_current),
                "future_values": self.future_values[index], "future_mask": self.future_masks[index],
                "selected_dates": self.selected_dates[index], "valid_months": int(self.valid_months[index]),
                "guide_available_date": str(self.available_dates[index]),
                "direction": self.directions[index], "bin_start_m": float(self.bin_starts[index]),
                "bin_end_m": float(self.bin_ends[index]), "distance_difference_m": float(distance[index]),
                "spatially_near": bool(near[index]), "temporal_condition_applied": bool(temporal_applied[index]),
                "similarity": float(best[int(offset)]), "cutoff_maintenance_date": self.cutoffs[index],
                "maintenance_type": self.maintenance_types[index],
                "maintenance_description": self.maintenance_descriptions[index],
            })
            if len(selected) >= config.top_k:
                break
        return selected

