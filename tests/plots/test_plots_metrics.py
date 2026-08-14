"""Unit tests for visualization functions in xwhy.plots.metrics."""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xwhy.metrics.regression import RegressionMetricResult
from xwhy.plots.metrics import (
    plot_fidelity,
    plot_importance_roc_curve,
    plot_stability_visualization,
)


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


class TestPlotStabilityVisualization:
    """Test suite for the plot_stability_visualization function."""

    @patch("xwhy.plots.metrics.plt.show")
    @patch("xwhy.plots.metrics.plt.savefig")
    def test_plot_stability_visualization_success(
        self,
        mock_savefig: MagicMock,
        mock_show: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test successful stability visualization generation and saving."""
        result_one = MagicMock()
        result_one.feature_names = ["word1", "word2"]
        result_one.coefficients = np.array([0.5, -0.5])

        result_two = MagicMock()
        result_two.feature_names = ["word3", "word4"]
        result_two.coefficients = np.array([0.2, -0.8])

        save_path = str(tmp_path / "stability.png")

        plot_stability_visualization(
            result_one,
            result_two,
            width=10.0,
            height=6.0,
            save_path=save_path,
        )

        mock_savefig.assert_called_once()
        mock_show.assert_called_once()

    @patch("xwhy.plots.metrics.plt.show")
    def test_plot_stability_visualization_zero_denom(
        self,
        mock_show: MagicMock,
    ) -> None:
        """Test stability visualization handles zero coefficients (denom == 0)."""
        result_one = MagicMock()
        result_one.feature_names = ["word1"]
        result_one.coefficients = np.array([0.0])

        result_two = MagicMock()
        result_two.feature_names = ["word2"]
        result_two.coefficients = np.array([0.0])

        plot_stability_visualization(result_one, result_two)

        mock_show.assert_called_once()


class TestPlotImportanceRocCurve:
    """Test suite for the plot_importance_roc_curve function."""

    def test_missing_truth_raises_value_error(self) -> None:
        """Ensure ValueError is raised when truth parameter is missing."""
        result = MagicMock()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "The 'truth' parameter (list of ints) is required in kwargs."
            ),
        ):
            plot_importance_roc_curve(result)

    def test_mismatched_length_raises_value_error(self) -> None:
        """Ensure ValueError is raised on truth labels and scores length mismatch."""
        result = MagicMock()
        result.coefficients = np.array([0.1, 0.2, 0.3])

        with pytest.raises(
            ValueError,
            match="Length of truth labels and result scores must match",
        ):
            plot_importance_roc_curve(result, truth=[1, 0])

    @patch("xwhy.plots.metrics.logger.warning")
    @patch("xwhy.plots.metrics.plt.show")
    def test_single_class_truth_logs_warning(
        self,
        mock_show: MagicMock,
        mock_logger_warning: MagicMock,
    ) -> None:
        """Log warning and skip plotting when only one class is present in truth."""
        result = MagicMock()
        result.coefficients = np.array([0.1, 0.2, 0.3])

        plot_importance_roc_curve(result, truth=[1, 1, 1])

        mock_logger_warning.assert_called_once()
        mock_show.assert_not_called()

    @patch("xwhy.plots.metrics.plt.show")
    @patch("xwhy.plots.metrics.plt.savefig")
    def test_plot_importance_roc_curve_success(
        self,
        mock_savefig: MagicMock,
        mock_show: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test successful ROC curve plotting and saving."""
        result = MagicMock()
        result.coefficients = np.array([0.1, 0.8, 0.3, 0.9])
        save_path = str(tmp_path / "roc_curve.png")

        plot_importance_roc_curve(
            result,
            truth=[0, 1, 0, 1],
            title="Custom ROC Title",
            save_path=save_path,
        )

        mock_savefig.assert_called_once()
        mock_show.assert_called_once()

    @patch("xwhy.plots.metrics.logger.debug")
    @patch("xwhy.plots.metrics.plt.show")
    @patch("xwhy.plots.metrics.plt.savefig")
    def test_plot_importance_roc_curve_with_save_path_and_logging(
        self,
        mock_savefig: MagicMock,
        mock_show: MagicMock,
        mock_logger_debug: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test ROC curve saves to disk and logs the debug message."""
        result = MagicMock()
        result.coefficients = np.array([0.1, 0.8, 0.3, 0.9])
        save_path = str(tmp_path / "roc_curve.png")

        plot_importance_roc_curve(
            result,
            truth=[0, 1, 0, 1],
            save_path=save_path,
        )

        mock_savefig.assert_called_once()
        mock_logger_debug.assert_called_once_with("ROC curve saved to: %s", save_path)
        mock_show.assert_called_once()

    @patch("xwhy.plots.metrics.plt.show")
    def test_plot_importance_roc_curve_without_save_path(
        self,
        mock_show: MagicMock,
    ) -> None:
        """Test ROC curve displays correctly without saving to disk."""
        result = MagicMock()
        result.coefficients = np.array([0.1, 0.8, 0.3, 0.9])

        plot_importance_roc_curve(
            result,
            truth=[0, 1, 0, 1],
        )

        mock_show.assert_called_once()
