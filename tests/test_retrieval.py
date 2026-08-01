import json
import sqlite3

import numpy as np

from acceleration_forecasting_12m.retrieval.build import _embedding_status, _truncate_future
from acceleration_forecasting_12m.retrieval.database import SCHEMA
from acceleration_forecasting_12m.retrieval.search import GuideIndex, SearchConfig


def source_row(valid=8):
    values = [float(i) if i < valid else None for i in range(18)]
    masks = [1 if i < valid else 0 for i in range(18)]
    dates = [f"d{i}" if i < valid else None for i in range(18)]
    return ("trend", "D_2000-2100_001", "2023-01-01", "D", 2000, 2100, 1.0,
            json.dumps(values), json.dumps(masks), json.dumps(dates), None, None, None)


def test_guide_validity_boundary_and_twelve_month_truncation():
    parsed, status, valid = _truncate_future(source_row(8), 12, 8)
    assert status == "eligible" and valid == 8
    assert len(parsed[1]) == len(parsed[2]) == len(parsed[3]) == 12
    assert _truncate_future(source_row(7), 12, 8)[1] == "insufficient_valid_months"


def test_embedding_must_be_finite_256d_and_nonzero():
    assert _embedding_status(np.ones(256, np.float32).tobytes(), 256, 256) == "valid"
    assert _embedding_status(np.zeros(256, np.float32).tobytes(), 256, 256) == "zero_norm_embedding"
    assert _embedding_status(np.ones(10, np.float32).tobytes(), 10, 256) == "invalid_embedding"


def test_search_excludes_same_dataset_and_selects_distinct_dates():
    connection = sqlite3.connect(":memory:"); connection.executescript(SCHEMA)
    embedding = np.ones(256, np.float32); embedding /= np.linalg.norm(embedding)
    for index, (dataset, date, start) in enumerate([
        ("D_2000-2100_001", "2022-01-01", 2000),
        ("D_3000-3100_001", "2022-02-01", 3000),
        ("U_4000-4100_001", "2022-03-01", 4000),
        ("D_5000-5100_001", "2022-04-01", 5000),
    ]):
        trend = f"t{index}"; record = f"r{index}"
        connection.execute(
            "INSERT INTO trends VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (trend, dataset, "model_train", date, dataset[0], start, start + 100, 1.1,
             json.dumps([1.0] * 12), json.dumps([1] * 12), json.dumps([date] * 12),
             "2023-04-16", None, None, None),
        )
        connection.execute(
            "INSERT INTO waveform_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (record, record, date, dataset[0], start, start + 100, 60, "source", record,
             dataset, trend, embedding.tobytes(), 256),
        )
    index = GuideIndex(connection)
    found = index.search(embedding, query_date="2025-01-01", query_dataset_id="D_2000-2100_001",
                         query_current=1.0, query_bin_start_m=2000,
                         allowed_splits={"model_train"}, config=SearchConfig())
    assert len(found) == 3
    assert all(item["dataset_id"] != "D_2000-2100_001" for item in found)
    assert len({item["measurement_date"] for item in found}) == 3


def test_near_candidate_requires_completed_past_but_far_candidate_does_not():
    connection = sqlite3.connect(":memory:"); connection.executescript(SCHEMA)
    embedding = np.ones(256, np.float32); embedding /= np.linalg.norm(embedding)
    for index, (dataset, date, start, available) in enumerate([
        ("D_2100-2200_002", "2023-01-01", 2100.0000004, "2025-01-01"),
        ("D_2200-2300_002", "2023-02-01", 2200.0001, "2025-01-01"),
    ]):
        trend = f"near_far_t{index}"; record = f"near_far_r{index}"
        connection.execute(
            "INSERT INTO trends VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (trend, dataset, "model_train", date, "D", start, start + 100, 1.0,
             json.dumps([1.0] * 12), json.dumps([1] * 12), json.dumps([date] * 12),
             available, None, None, None),
        )
        connection.execute(
            "INSERT INTO waveform_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (record, record, date, "D", start, start + 100, 60, "source", record,
             dataset, trend, embedding.tobytes(), 256),
        )
    found = GuideIndex(connection).search(
        embedding, query_date="2024-01-01", query_dataset_id="D_2000-2100_001",
        query_current=1.0, query_bin_start_m=2000.0,
        allowed_splits={"model_train"}, config=SearchConfig(),
    )
    assert [item["dataset_id"] for item in found] == ["D_2200-2300_002"]
    assert found[0]["spatially_near"] is False
    assert found[0]["temporal_condition_applied"] is False
