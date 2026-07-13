"""Unit tests for visualization functions in xwhy.plots.metrics."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xwhy.metrics.regression import RegressionMetricResult
from xwhy.plots.metrics import plot_fidelity


@pytest.fixture
def dummy_metrics() -> RegressionMetricResult:
    """Fixture providing dummy regression metrics for testing."""
    return RegressionMetricResult(
        weighted_mse=0.1,
        weighted_mae=0.2,
        weighted_r2=0.85,
        weighted_adj_r2=0.80,
        mean_loss=0.15,
        mean_l1_loss=0.1,
        mean_l2_loss=0.2,
        weighted_l1_norm=0.1,
        weighted_l2_norm=0.2,
    )


class TestPlotFidelity:
    """Test suite for the plot_fidelity function."""

    def test_mismatched_array_lengths(
        self, dummy_metrics: RegressionMetricResult
    ) -> None:
        """Ensure ValueError is raised when array lengths do not match."""
        y_target = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9])  # Mismatched length
        weights = np.array([0.5, 0.5, 0.5])

        with pytest.raises(ValueError, match="same length"):
            plot_fidelity(
                metrics=dummy_metrics,
                y_target=y_target,
                y_pred=y_pred,
                weights=weights,
            )

    @patch("xwhy.plots.metrics.plt.show")
    @patch("xwhy.plots.metrics.plt.savefig")
    def test_plot_fidelity_success_without_save(
        self,
        mock_savefig: MagicMock,
        mock_show: MagicMock,
        dummy_metrics: RegressionMetricResult,
    ) -> None:
        """Test successful plot generation without saving to disk."""
        y_target = np.array([1.0, 2.0])
        y_pred = np.array([1.1, 1.9])
        weights = np.array([1.0, 0.5])

        result = plot_fidelity(
            metrics=dummy_metrics,
            y_target=y_target,
            y_pred=y_pred,
            weights=weights,
            save_path=None,
            show=True,
        )

        assert result is None
        mock_show.assert_called_once()
        mock_savefig.assert_not_called()

    @patch("xwhy.plots.metrics.plt.show")
    def test_plot_fidelity_with_save_path(
        self,
        mock_show: MagicMock,
        dummy_metrics: RegressionMetricResult,
        tmp_path: Path,
    ) -> None:
        """Test plotting and saving functionality using a temporary directory."""
        y_target = np.array([1.0, 2.0])
        y_pred = np.array([1.1, 1.9])
        weights = np.array([1.0, 0.5])

        save_path = tmp_path / "output" / "test_plot.png"

        result = plot_fidelity(
            metrics=dummy_metrics,
            y_target=y_target,
            y_pred=y_pred,
            weights=weights,
            save_path=save_path,
            show=False,
        )

        assert result is not None
        # Ensure the path returned is absolute and matches our intention
        assert str(save_path.resolve()) == result
        # Check if the file was actually created by matplotlib
        assert Path(result).exists()
        mock_show.assert_not_called()

    @patch("xwhy.plots.metrics.plt.show")
    def test_plot_fidelity_zero_weights_edge_case(
        self,
        mock_show: MagicMock,
        dummy_metrics: RegressionMetricResult,
    ) -> None:
        """Ensure the function handles max_weight == 0 gracefully."""
        y_target = np.array([1.0, 2.0])
        y_pred = np.array([1.1, 1.9])
        weights = np.array([0.0, 0.0])  # Edge case: max weight is zero

        # Should not raise any division by zero errors
        result = plot_fidelity(
            metrics=dummy_metrics,
            y_target=y_target,
            y_pred=y_pred,
            weights=weights,
            show=False,
        )

        assert result is None
        mock_show.assert_not_called()
