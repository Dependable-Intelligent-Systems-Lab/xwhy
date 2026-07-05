"""Unit tests for metrics models."""

import numpy as np

from xwhy.metrics.regression import RegressionMetrics


def test_regression_metrics() -> None:
    """Test regression metric calculations."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    weights = np.ones(3)

    result = RegressionMetrics.calculate(
        y_true=y_true, y_pred=y_pred, weights=weights, num_features=1
    )

    assert result.weighted_mse == 0.0
    assert result.weighted_r2 == 1.0
