from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from acceleration_forecasting_12m.common.constants import (
    FORECAST_MONTHS, HISTORY_MONTHS, MIN_GUIDE_MONTHS, MIN_HISTORY_MONTHS,
    MIN_TARGET_MONTHS, TOP_K,
)
from acceleration_forecasting_12m.common.io import artifact_descriptor, write_json
from acceleration_forecasting_12m.common.progress import progress_bar
from acceleration_forecasting_12m.retrieval.autoencoder import WaveformAutoencoder, normalize_embeddings
from acceleration_forecasting_12m.retrieval.database import connect_read_only
from acceleration_forecasting_12m.retrieval.search import GuideIndex, SearchConfig
from .baseline import softmax_guide_baseline
from .history import select_monthly_history, truncate_future
from .normalization import Normalization


ARRAY_NAMES = (
    "current_values", "history_values", "history_masks", "guide_values", "guide_masks",
    "guide_deltas", "guide_similarities", "retrieval_masks", "guide_baselines", "guide_softmax_weights",
)


def _prepare_frame(frame, target_column):
    frame = frame.copy()
    frame["measurement_date"] = pd.to_datetime(frame["measurement_date"], errors="raise").dt.normalize()
    frame["current_acc_z_max"] = pd.to_numeric(frame["current_acc_z_max"], errors="coerce")
    frame["target_id"] = frame[target_column].astype(str)
    return frame


def _load_frames(source_dir):
    manifest = pd.read_csv(source_dir / "dataset_split_manifest.csv", encoding="utf-8-sig")
    development = pd.read_csv(source_dir / "development_trends.csv", encoding="utf-8-sig")
    inference = pd.read_csv(source_dir / "inference_targets.csv", encoding="utf-8-sig")
    split_map = manifest.drop_duplicates("dataset_id").set_index("dataset_id")["model_split"]
    development["model_split"] = development["dataset_id"].map(split_map)
    inference["model_split"] = "inference"
    return manifest, _prepare_frame(development, "trend_id"), _prepare_frame(inference, "target_id")


def _development_embeddings(connection):
    output = defaultdict(list)
    for record_id, trend_id, blob, dimension in connection.execute(
        "SELECT record_id,trend_id,embedding,embedding_dim FROM waveform_records"
    ):
        vector = np.frombuffer(blob, dtype=np.float32, count=int(dimension)).copy()
        output[str(trend_id)].append((str(record_id), vector))
    return output


def _source_waveform_info(source_dir):
    source = json.loads((source_dir / "source_artifacts.json").read_text(encoding="utf-8"))
    waveform_path = Path(source["waveforms_path"])
    return source, waveform_path


def _encode_inference(rows, source_dir, device, batch_size=512, progress=True):
    if rows.empty:
        return {}
    checkpoint = torch.load(source_dir / "autoencoder.pt", map_location="cpu", weights_only=False)
    model = WaveformAutoencoder(checkpoint.get("embedding_dim", 256))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)
    source, waveform_path = _source_waveform_info(source_dir)
    waveforms = np.memmap(
        waveform_path, mode="r", dtype=np.float32,
        shape=(int(source["record_count"]), int(source["samples_per_waveform"])),
    )
    mean, std = float(checkpoint["mean"]), float(checkpoint["std"])
    output = {}
    records = list(rows.itertuples(index=False))
    with torch.inference_mode():
        batches = range(0, len(records), int(batch_size))
        for start in progress_bar(batches, enabled=progress, total=(len(records) + batch_size - 1) // batch_size,
                                  desc="inference波形を256次元化", unit="batch"):
            batch = records[start:start + batch_size]
            values = np.stack([waveforms[int(item.waveform_index)] for item in batch]).copy()
            tensor = torch.from_numpy((values - mean) / std).unsqueeze(1).to(device)
            embedded = normalize_embeddings(model.encode(tensor)).cpu().numpy().astype(np.float32)
            for item, vector in zip(batch, embedded):
                output[str(item.record_id)] = vector
    return output


def _select_anchors(frame, split, groups, has_query, *, min_history, min_target, limit=None):
    selected, reasons = [], Counter()
    ordered = frame.sort_values(["dataset_id", "measurement_date", "target_id"], kind="mergesort")
    for dataset_id, candidates in ordered.groupby("dataset_id", sort=False):
        if limit is not None and len(selected) >= int(limit):
            break
        adopted = None
        for _, row in candidates.iterrows():
            current = float(row["current_acc_z_max"])
            if not np.isfinite(current):
                reasons["non_finite_current"] += 1; continue
            history, history_mask, history_dates = select_monthly_history(
                groups[str(dataset_id)], row["measurement_date"], months=HISTORY_MONTHS
            )
            if int(history_mask.sum()) < int(min_history):
                reasons["insufficient_history"] += 1; continue
            future, future_mask, future_dates = truncate_future(row, months=FORECAST_MONTHS)
            if split != "inference" and int(future_mask.sum()) < int(min_target):
                reasons["insufficient_target"] += 1; continue
            if not has_query(split, str(row["target_id"])):
                reasons["missing_query_waveform"] += 1; continue
            adopted = {
                "row": row, "current": current, "history": history, "history_mask": history_mask,
                "history_dates": history_dates, "future": future, "future_mask": future_mask,
                "future_dates": future_dates,
            }
            break
        if adopted is None:
            reasons["dataset_without_anchor"] += 1
        else:
            selected.append(adopted)
    return selected, {str(key): int(value) for key, value in reasons.items()}


def _save_inputs(path, metadata, arrays):
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metadata).to_csv(path / "metadata.csv", index=False, encoding="utf-8-sig")
    for name in ARRAY_NAMES:
        np.save(path / f"{name}.npy", np.asarray(arrays[name], dtype=np.float32))


def _construct_split(selected, split, index, query_lookup, allowed_splits, search_config,
                     temperature, assignments, progress=True):
    metadata, arrays = [], {name: [] for name in ARRAY_NAMES}
    targets, target_masks, target_dates = [], [], []
    for anchor in progress_bar(selected, enabled=progress, total=len(selected),
                               desc=f"{split}データを構築", unit="target"):
        row, current = anchor["row"], anchor["current"]
        query_vectors = query_lookup(split, str(row["target_id"]))
        guides = index.search(
            np.stack(query_vectors), query_date=pd.Timestamp(row["measurement_date"]).strftime("%Y-%m-%d"),
            query_dataset_id=str(row["dataset_id"]), query_current=current,
            query_bin_start_m=float(row["bin_start_m"]), allowed_splits=allowed_splits,
            config=search_config,
        )
        guide_values = np.full((TOP_K, FORECAST_MONTHS), np.nan, np.float32)
        guide_masks = np.zeros((TOP_K, FORECAST_MONTHS), np.float32)
        guide_deltas = np.zeros((TOP_K, FORECAST_MONTHS), np.float32)
        similarities = np.zeros(TOP_K, np.float32)
        retrieval_masks = np.zeros(TOP_K, np.float32)
        for rank, guide in enumerate(guides):
            values = np.asarray([np.nan if value is None else value for value in guide["future_values"]], np.float32)
            mask = np.asarray(guide["future_mask"], np.float32) * np.isfinite(values)
            guide_values[rank], guide_masks[rank] = values, mask
            guide_deltas[rank] = np.where(mask > 0, values - float(guide["current_acc_z_max"]), np.nan)
            similarities[rank], retrieval_masks[rank] = guide["similarity"], 1
            assignments.append({
                "split": split, "target_id": str(row["target_id"]), "guide_rank": rank + 1,
                "selection_status": "selected", "query_date": row["measurement_date"].strftime("%Y-%m-%d"),
                "query_dataset_id": row["dataset_id"], "guide_record_id": guide["record_id"],
                "guide_trend_id": guide["trend_id"], "guide_date": guide["measurement_date"],
                "candidate_dataset_id": guide["dataset_id"], "candidate_direction": guide["direction"],
                "candidate_bin_start_m": guide["bin_start_m"], "candidate_bin_end_m": guide["bin_end_m"],
                "cosine_similarity": guide["similarity"], "current_max_difference": guide["current_max_difference"],
                "guide_valid_months": guide["valid_months"], "guide_available_date": guide["guide_available_date"],
                "distance_difference_m": guide["distance_difference_m"],
                "spatially_near": guide["spatially_near"],
                "temporal_condition_applied": guide["temporal_condition_applied"],
            })
        for rank in range(len(guides), TOP_K):
            assignments.append({
                "split": split, "target_id": str(row["target_id"]), "guide_rank": rank + 1,
                "selection_status": "not_found", "query_date": row["measurement_date"].strftime("%Y-%m-%d"),
                "query_dataset_id": row["dataset_id"],
            })
        baseline, weights = softmax_guide_baseline(
            guide_values, guide_masks, similarities, retrieval_masks, current, temperature
        )
        metadata.append({
            "target_id": str(row["target_id"]), "dataset_id": str(row["dataset_id"]),
            "anchor_date": row["measurement_date"].strftime("%Y-%m-%d"),
            "current_acc_z_max": current, "direction": row["direction"],
            "bin_start_m": float(row["bin_start_m"]), "bin_end_m": float(row["bin_end_m"]),
            "history_dates": json.dumps(anchor["history_dates"]),
            "valid_history_months": int(anchor["history_mask"].sum()),
            "valid_target_months": int(anchor["future_mask"].sum()), "guide_count": len(guides),
            "cutoff_maintenance_date": row.get("cutoff_maintenance_date", "") if pd.notna(row.get("cutoff_maintenance_date", "")) else "",
            "maintenance_type": row.get("maintenance_type", "") if pd.notna(row.get("maintenance_type", "")) else "",
            "maintenance_description": row.get("maintenance_description", "") if pd.notna(row.get("maintenance_description", "")) else "",
        })
        for name, value in {
            "current_values": [current], "history_values": anchor["history"],
            "history_masks": anchor["history_mask"], "guide_values": guide_values,
            "guide_masks": guide_masks, "guide_deltas": guide_deltas, "guide_similarities": similarities,
            "retrieval_masks": retrieval_masks, "guide_baselines": baseline,
            "guide_softmax_weights": weights,
        }.items():
            arrays[name].append(value)
        targets.append(anchor["future"]); target_masks.append(anchor["future_mask"])
        target_dates.append(anchor["future_dates"])
    return metadata, arrays, np.asarray(targets, np.float32), np.asarray(target_masks, np.float32), target_dates


def prepare_datasets(source_artifact_dir, retrieval_dir, output_dir, *, device=None,
                     min_history_months=MIN_HISTORY_MONTHS, min_target_months=MIN_TARGET_MONTHS,
                     min_guide_months=MIN_GUIDE_MONTHS, temperature=0.1,
                     max_train=None, max_validation=None, max_inference=None, progress=True,
                     target_mode="residual"):
    source_dir, retrieval_dir, output_dir = map(lambda value: Path(value).resolve(),
                                                (source_artifact_dir, retrieval_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, development, inference = _load_frames(source_dir)
    all_trends = pd.concat([development, inference], ignore_index=True)
    groups = {str(key): group.copy() for key, group in all_trends.groupby("dataset_id", sort=False)}
    connection = connect_read_only(retrieval_dir / "vector_database_12m.sqlite")
    index = GuideIndex(connection)
    development_vectors = _development_embeddings(connection)
    connection.close()
    inference_inputs = pd.read_csv(source_dir / "inference_inputs.csv", encoding="utf-8-sig")
    inference_rows = {
        str(target_id): group.copy() for target_id, group in inference_inputs.groupby("target_id", sort=False)
    }

    def has_query(split, target_id):
        return bool(inference_rows.get(target_id) is not None) if split == "inference" else bool(development_vectors.get(target_id))

    specifications = (
        ("model_train", development.loc[development["model_split"] == "model_train"], max_train),
        ("model_validation", development.loc[development["model_split"] == "model_validation"], max_validation),
        ("inference", inference, max_inference),
    )
    selected, selection_diagnostics = {}, {}
    for split, frame, limit in specifications:
        selected[split], selection_diagnostics[split] = _select_anchors(
            frame, split, groups, has_query, min_history=min_history_months,
            min_target=min_target_months, limit=limit,
        )
    wanted_inference_ids = {str(item["row"]["target_id"]) for item in selected["inference"]}
    inference_to_encode = inference_inputs.loc[inference_inputs["target_id"].astype(str).isin(wanted_inference_ids)]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    inference_vectors = _encode_inference(inference_to_encode, source_dir, device, progress=progress)

    def query_lookup(split, target_id):
        if split != "inference":
            return [vector for _, vector in development_vectors.get(target_id, [])]
        rows = inference_rows.get(target_id)
        if rows is None:
            return []
        return [inference_vectors[str(record_id)] for record_id in rows["record_id"] if str(record_id) in inference_vectors]

    search_config = SearchConfig(min_valid_months=int(min_guide_months))
    assignments, built = [], {}
    for split, _, _ in specifications:
        allowed = {"model_train"} if split != "inference" else {"model_train", "model_validation"}
        built[split] = _construct_split(
            selected[split], split, index, query_lookup, allowed, search_config,
            float(temperature), assignments, progress,
        )

    train_metadata, train_arrays, train_target, train_mask, _ = built["model_train"]
    condition_values = [np.asarray(train_arrays["current_values"]).ravel()]
    history = np.asarray(train_arrays["history_values"]); history_mask = np.asarray(train_arrays["history_masks"])
    condition_values.append(history[history_mask > 0])
    condition_norm = Normalization.fit(np.concatenate(condition_values), "model_train_current_and_history")
    if target_mode not in {"residual", "absolute"}:
        raise ValueError("target_mode must be residual or absolute")
    train_baseline = np.asarray(train_arrays["guide_baselines"])
    training_values = train_target - train_baseline if target_mode == "residual" else train_target
    target_norm = Normalization.fit(training_values[train_mask > 0], f"model_train_{target_mode}")
    valid_training = training_values[(train_mask > 0) & np.isfinite(training_values)]
    low, high = np.percentile(valid_training, [0.5, 99.5]); radius = float(max(abs(low), abs(high)))

    for split in ("model_train", "model_validation", "inference"):
        metadata, arrays, targets, masks, dates = built[split]
        destination = output_dir / split if split != "inference" else output_dir / "inference" / "inputs"
        _save_inputs(destination, metadata, arrays)
        if split != "inference":
            values = targets - np.asarray(arrays["guide_baselines"], np.float32) if target_mode == "residual" else targets
            np.save(destination / "target_values.npy", values.astype(np.float32))
            np.save(destination / "target_masks.npy", masks.astype(np.float32))
        else:
            target_dir = output_dir / "inference" / "targets"; target_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"target_id": [item["target_id"] for item in metadata]}).to_csv(
                target_dir / "target_ids.csv", index=False, encoding="utf-8-sig"
            )
            np.save(target_dir / "target_values.npy", targets.astype(np.float32))
            np.save(target_dir / "target_masks.npy", masks.astype(np.float32))
            (target_dir / "selected_dates.json").write_text(json.dumps(dates), encoding="utf-8")
    condition_norm.save(output_dir / "condition_normalization.json")
    target_norm.save(output_dir / "normalization.json")
    pd.DataFrame(assignments).to_csv(output_dir / "guide_assignments.csv", index=False, encoding="utf-8-sig")
    identity = {
        "history_months": HISTORY_MONTHS, "min_history_months": int(min_history_months),
        "forecast_months": FORECAST_MONTHS, "min_target_months": int(min_target_months),
        "min_guide_months": int(min_guide_months), "temperature": float(temperature),
        "target_mode": target_mode,
    }
    build_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    guide_config = {
        **identity, "dataset_build_id": build_id, "top_k": TOP_K,
        "max_current_difference": 0.5, "near_distance_m": 100.0,
        "sampling_clip_physical": [-radius, radius] if target_mode == "residual" else [0.1, 6.0],
        "residual_clip_physical": [-radius, radius], "final_physical_bounds": [0.1, 6.0],
    }
    write_json(output_dir / "guide_search_config.json", guide_config)
    source_artifacts = {
        "retrieval_database": artifact_descriptor(retrieval_dir / "vector_database_12m.sqlite"),
        "split_manifest": artifact_descriptor(source_dir / "dataset_split_manifest.csv"),
        "autoencoder": artifact_descriptor(source_dir / "autoencoder.pt"),
        "source_artifact_dir": str(source_dir),
    }
    write_json(output_dir / "source_artifacts.json", source_artifacts)
    summary = {
        "dataset_build_id": build_id,
        "counts": {split: len(built[split][0]) for split in built},
        "selection_diagnostics": selection_diagnostics,
        "history_months": HISTORY_MONTHS, "min_history_months": int(min_history_months),
        "forecast_months": FORECAST_MONTHS, "min_target_months": int(min_target_months),
        "min_guide_months": int(min_guide_months), "target_mode": target_mode,
        "sampling_clip_physical": guide_config["sampling_clip_physical"],
    }
    write_json(output_dir / "dataset_summary.json", summary)
    return summary
