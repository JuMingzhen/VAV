import math

import numpy as np

from vav.analysis import attention_entropy, bbox_overlap, normalize_map, topk_patch_concentration


def test_normalize_map_returns_unit_range():
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    normalized = normalize_map(values)
    assert np.isclose(normalized.min(), 0.0)
    assert np.isclose(normalized.max(), 1.0)


def test_attention_entropy_prefers_uniform_heatmap():
    sharp = np.array([[1.0, 0.0], [0.0, 0.0]])
    uniform = np.ones((2, 2))
    assert attention_entropy(uniform) > attention_entropy(sharp)


def test_topk_patch_concentration_detects_peaked_attention():
    peaked = np.array([[10.0, 0.0], [0.0, 0.0]])
    spread = np.ones((2, 2))
    assert topk_patch_concentration(peaked, k=1) > topk_patch_concentration(spread, k=1)


def test_bbox_overlap_handles_missing_box():
    value = bbox_overlap(np.ones((2, 2)), None, (100, 100))
    assert math.isnan(value)
