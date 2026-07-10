"""Unit tests for metrics models."""

import dataclasses

import numpy as np
import pytest

from xwhy.metrics import RegressionMetricResult, RegressionMetrics


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


def test_regression_metric_result_initialization() -> None:
    """Test data initialization and immutability (frozen)."""
    metrics = RegressionMetricResult(
        weighted_mse=0.12345,
        weighted_mae=0.23456,
        weighted_r2=0.8500,
        weighted_adj_r2=0.8400,
        mean_loss=0.5,
        mean_l1_loss=0.6,
        mean_l2_loss=0.7,
        weighted_l1_norm=0.8,
        weighted_l2_norm=0.9,
    )

    # تست مقادیر
    assert metrics.weighted_mse == 0.12345
    assert metrics.weighted_r2 == 0.8500

    with pytest.raises(dataclasses.FrozenInstanceError):
        metrics.weighted_mse = 0.0  # type: ignore


def test_regression_metric_result_str() -> None:
    """Test the string representation formatting."""
    metrics = RegressionMetricResult(
        weighted_mse=0.1,
        weighted_mae=0.2,
        weighted_r2=0.9,
        weighted_adj_r2=0.88,
        mean_loss=0.3,
        mean_l1_loss=0.4,
        mean_l2_loss=0.5,
        weighted_l1_norm=0.6,
        weighted_l2_norm=0.7,
    )

    str_output = str(metrics)

    assert "Fidelity Metrics:" in str_output
    assert "Mean Squared Error (MSE)            0.1000" in str_output
    assert "Mean Absolute Error (MAE)           0.2000" in str_output
    assert "Mean Loss (Lm)                      0.3000" in str_output
    assert "Mean L1 Loss                        0.4000" in str_output
    assert "Mean L2 Loss                        0.5000" in str_output
    assert "Weighted L1 Loss                    0.6000" in str_output
    assert "Weighted L2 Loss                    0.7000" in str_output
    assert "Weighted R-squared (R²ω)            0.9000" in str_output
    assert "Weighted Adjusted R-squared (R^²ω)  0.8800" in str_output

    assert "-" * 80 in str_output
