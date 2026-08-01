import numpy as np
import pandas as pd

from acceleration_forecasting_12m.datasets.baseline import softmax_guide_baseline
from acceleration_forecasting_12m.datasets.history import select_monthly_history, truncate_future


def test_history_uses_five_prior_months_without_anchor_overlap():
    frame = pd.DataFrame({
        "measurement_date": pd.to_datetime([
            "2024-01-01", "2024-02-02", "2024-03-01", "2024-04-02", "2024-05-01", "2024-06-17"
        ]),
        "current_acc_z_max": [1, 2, 3, 4, 5, 99],
    })
    values, mask, dates = select_monthly_history(frame, "2024-06-17")
    assert values.tolist() == [1, 2, 3, 4, 5]
    assert mask.tolist() == [1, 1, 1, 1, 1]
    assert "2024-06-17" not in dates


def test_history_acceptance_boundary_is_three_of_five():
    frame = pd.DataFrame({
        "measurement_date": pd.to_datetime(["2024-01-01", "2024-03-01", "2024-05-01"]),
        "current_acc_z_max": [1, 3, 5],
    })
    _, mask, _ = select_monthly_history(frame, "2024-06-17")
    assert int(mask.sum()) == 3


def test_truncate_future_returns_first_twelve_months():
    row = {
        "measurement_date": "2024-01-10", "future_values": list(range(18)),
        "future_mask": [1] * 18, "selected_dates": [f"d{i}" for i in range(18)],
        "cutoff_maintenance_date": "",
    }
    values, mask, dates = truncate_future(row)
    assert values.shape == mask.shape == (12,)
    assert values[-1] == 11 and dates[-1] == "d11"


def test_masked_softmax_baseline_uses_only_valid_guides_and_fills_gaps():
    values = np.array([[1, np.nan, 3], [2, 2, np.nan]], dtype=float)
    masks = np.isfinite(values).astype(float)
    baseline, weights = softmax_guide_baseline(values, masks, [1, 0], [1, 1], 9, temperature=0.1)
    assert baseline.shape == (3,)
    assert np.isfinite(baseline).all()
    assert baseline[1] == 2
    assert weights[0, 0] > weights[1, 0]

