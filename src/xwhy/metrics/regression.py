"""Regression metrics calculation."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class RegressionMetricResult:
    """Data container for regression evaluation metrics."""

    weighted_mse: float
    weighted_mae: float
    weighted_r2: float
    weighted_adj_r2: float
    mean_loss: float
    mean_l1_loss: float
    mean_l2_loss: float
    weighted_l1_norm: float
    weighted_l2_norm: float

    def __str__(self) -> str:
        """Representation of metrics for easy printing."""
        lines = [
            "-" * 80,
            "Fidelity Metrics:",
            f"  Weighted MSE:     {self.weighted_mse:.4f}",
            f"  Weighted MAE:     {self.weighted_mae:.4f}",
            f"  Weighted R²:      {self.weighted_r2:.4f}",
            f"  Adj Weighted R²:  {self.weighted_adj_r2:.4f}",
            f"  Mean Loss:        {self.mean_loss:.4f}",
            "-" * 80,
        ]
        return "\n".join(lines)


class RegressionMetrics:
    """Utility for calculating comprehensive regression metrics."""

    @classmethod
    def calculate(
        cls,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: np.ndarray,
        num_features: int,
    ) -> RegressionMetricResult:
        """Compute a comprehensive suite of regression metrics.

        Args:
            y_true: Array of true target values.
            y_pred: Array of predicted target values.
            weights: Array of sample weights.
            num_features: Number of features used in the model (for adjusted R2).

        Returns:
            RegressionMetricResult: An immutable dataclass containing all metrics.

        """
        mse = mean_squared_error(y_true, y_pred, sample_weight=weights)
        mae = mean_absolute_error(y_true, y_pred, sample_weight=weights)
        r2 = r2_score(y_true, y_pred, sample_weight=weights)

        n = len(y_true)
        diff = y_true - y_pred

        mean_loss = float(np.abs(np.mean(y_true) - np.mean(y_pred)))
        mean_l1 = float(np.mean(np.abs(diff)))
        mean_l2 = float(np.mean(diff**2))

        weighted_l1_norm_n = float(np.sum(weights * np.abs(diff)) / n)
        weighted_l2_norm_n = float(np.sum(weights * (diff**2)) / n)

        adj_r2 = float("nan")
        if n > num_features + 1:
            adj_r2 = float(1 - (1 - r2) * (n - 1) / (n - num_features - 1))

        return RegressionMetricResult(
            weighted_mse=float(mse),
            weighted_mae=float(mae),
            weighted_r2=float(r2),
            weighted_adj_r2=adj_r2,
            mean_loss=mean_loss,
            mean_l1_loss=mean_l1,
            mean_l2_loss=mean_l2,
            weighted_l1_norm=weighted_l1_norm_n,
            weighted_l2_norm=weighted_l2_norm_n,
        )
