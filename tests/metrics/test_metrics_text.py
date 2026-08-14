"""Tests for text evaluation metrics."""

from __future__ import annotations

import re

import numpy as np
import pytest

from xwhy.core.result import BaseXWhyResult
from xwhy.metrics.text import calculate_stability_score, calculate_token_auc


class DummyTextResult(BaseXWhyResult):
    """Dummy result class for testing text metrics."""

    def __init__(self, feature_names: list[str], coefficients: list[float]) -> None:
        """Initialize dummy text result."""
        self._feature_names = feature_names
        self._coefficients = coefficients

    @property
    def feature_names(self) -> list[str]:
        """Return mock feature names."""
        return self._feature_names

    @property
    def coefficients(self) -> list[float]:  # type: ignore[override]
        """Return mock coefficients."""
        return self._coefficients

    @property
    def data(self) -> np.ndarray:
        """Return empty mock data array."""
        return np.array([])


def test_calculate_stability_score_length_mismatch() -> None:
    """Test ValueError when feature and coefficient lengths mismatch."""
    res1 = DummyTextResult(["a", "b"], [1.0])
    res2 = DummyTextResult(["a"], [1.0])

    with pytest.raises(
        ValueError,
        match=re.escape("Length of features and contribution vectors must match."),
    ):
        calculate_stability_score(res1, res2)


def test_calculate_stability_score_zero_denominator() -> None:
    """Test stability score returns 1.0, 0.0 when both vectors are zero."""
    res1 = DummyTextResult(["a", "b"], [0.0, 0.0])
    res2 = DummyTextResult(["a", "b"], [0.0, 0.0])

    sim, dist = calculate_stability_score(res1, res2)
    assert sim == 1.0
    assert dist == 0.0


def test_calculate_stability_score_normal() -> None:
    """Test standard Generalized Jaccard similarity calculation."""
    res1 = DummyTextResult(["a", "b", "c"], [1.0, 2.0, 3.0])
    res2 = DummyTextResult(["b", "c", "d"], [2.0, 3.0, 4.0])

    sim, dist = calculate_stability_score(res1, res2)
    assert 0.0 <= sim <= 1.0
    assert abs((sim + dist) - 1.0) < 1e-6


def test_calculate_token_auc_length_mismatch() -> None:
    """Test ValueError when tokens, scores, and truth labels mismatch."""
    res = DummyTextResult(["a", "b"], [1.0, 2.0])
    truth = [1]

    with pytest.raises(ValueError, match="Dimension mismatch"):
        calculate_token_auc(res, truth)


def test_calculate_token_auc_single_class() -> None:
    """Test token AUC returns 0.5 when only one class is in truth."""
    res = DummyTextResult(["a", "b"], [1.0, 2.0])
    truth = [1, 1]

    score = calculate_token_auc(res, truth)
    assert score == 0.5


def test_calculate_token_auc_multiclass() -> None:
    """Test token AUC calculation with valid binary labels."""
    res = DummyTextResult(["a", "b", "c", "d"], [0.1, 0.9, 0.2, 0.8])
    truth = [0, 1, 0, 1]

    score = calculate_token_auc(res, truth)
    assert 0.0 <= score <= 1.0
