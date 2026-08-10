import numpy as np
import pytest

from acceleration_forecasting_12m.evaluation.visualize import monthly_histogram_mode


def test_monthly_histogram_mode_quantizes_each_month_and_prefers_larger_tie():
    samples = np.tile(np.arange(1, 13, dtype=float), (4, 1))
    samples[:, 0] = [1.01, 1.04, 1.16, 1.18]
    samples[:, 1] = [2.01, 2.04, 2.02, 2.14]

    result = monthly_histogram_mode(samples)

    assert result[0] == pytest.approx(1.2)  # 1.0 and 1.2 tie; prefer 1.2.
    assert result[1] == pytest.approx(2.0)  # 2.0 is the unique mode.
    assert np.allclose(result[2:], np.arange(3, 13, dtype=float))
    assert np.allclose(result * 10, np.rint(result * 10))


def test_monthly_histogram_mode_uses_half_up_boundaries_and_ignores_nan():
    samples = np.tile(np.arange(1, 13, dtype=float), (3, 1))
    samples[:, 0] = [1.15, 1.16, np.nan]

    result = monthly_histogram_mode(samples)

    assert result[0] == pytest.approx(1.2)


def test_monthly_histogram_mode_rejects_month_without_finite_values():
    samples = np.ones((3, 12), dtype=float)
    samples[:, 5] = np.nan

    with pytest.raises(ValueError, match="Month 6"):
        monthly_histogram_mode(samples)


def test_monthly_histogram_mode_requires_generation_by_twelve_month_shape():
    with pytest.raises(ValueError, match=r"\[generation, 12\]"):
        monthly_histogram_mode(np.ones((100, 11)))
