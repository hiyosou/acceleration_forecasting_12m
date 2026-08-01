from __future__ import annotations

import numpy as np


def softmax_guide_baseline(values, masks, similarities, retrieval_masks, current, temperature=0.1):
    values, masks = np.asarray(values, float), np.asarray(masks, float)
    similarities, retrieval_masks = np.asarray(similarities, float), np.asarray(retrieval_masks, float)
    if values.shape != masks.shape or values.ndim != 2:
        raise ValueError("Guide values and masks must have shape [guide, month]")
    if temperature <= 0:
        raise ValueError("Softmax temperature must be positive")
    valid = (masks > 0) & np.isfinite(values) & (retrieval_masks[:, None] > 0)
    baseline = np.full(values.shape[1], np.nan, dtype=float)
    weights = np.zeros_like(values, dtype=float)
    scores = similarities / float(temperature)
    for month in range(values.shape[1]):
        selected = valid[:, month]
        if not selected.any():
            continue
        logits = scores[selected] - np.max(scores[selected])
        month_weights = np.exp(logits); month_weights /= month_weights.sum()
        weights[selected, month] = month_weights
        baseline[month] = np.sum(month_weights * values[selected, month])
    finite = np.isfinite(baseline)
    if finite.any():
        indices = np.arange(len(baseline)); locations = indices[finite]
        baseline = np.interp(indices, locations, baseline[finite])
    else:
        baseline.fill(float(current))
    return baseline.astype(np.float32), weights.astype(np.float32)

