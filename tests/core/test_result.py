"""Unit tests for core results."""

from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xwhy.core.result import BaseXWhyResult, TextXWhyResult
from xwhy.metrics.regression import RegressionMetricResult


class ConcreteResult(BaseXWhyResult):
    """Concrete implementation of BaseXWhyResult for testing."""

    @property
    def feature_names(self) -> Sequence[str]:
        """Mock feature names for testing."""
        return ["feat1", "feat2"]

    @property
    def data(self) -> np.ndarray:
        """Mock data instance for testing."""
        return np.array([1, 2])


@pytest.fixture
def mock_metrics() -> RegressionMetricResult:
    """Fixture to provide a dummy metric result."""
    return RegressionMetricResult(
        weighted_mse=0.1,
        weighted_mae=0.1,
        weighted_r2=0.9,
        weighted_adj_r2=0.85,
        mean_loss=0.05,
        mean_l1_loss=0.1,
        mean_l2_loss=0.05,
        weighted_l1_norm=0.1,
        weighted_l2_norm=0.05,
    )


def test_base_result_initialization(mock_metrics: RegressionMetricResult) -> None:
    """Verify BaseXWhyResult initializes correctly."""
    coeffs = np.array([0.1, 0.2])
    result = ConcreteResult(coefficients=coeffs, metrics=mock_metrics)

    assert np.array_equal(result.coefficients, coeffs)
    assert result.metrics == mock_metrics
    assert result.raw_data == {}  # Check default factory
    assert result.base_values == 0.0


def test_text_result_initialization(mock_metrics: RegressionMetricResult) -> None:
    """Verify TextXWhyResult initializes with correct defaults."""
    coeffs = np.array([0.1, 0.2])
    words = ["test", "case"]
    result = TextXWhyResult(
        coefficients=coeffs,
        metrics=mock_metrics,
        original_output="test case",
        words=words,
    )

    assert result.original_output == "test case"
    assert result.words == words
    assert result.feature_names == words
    assert np.array_equal(result.data, np.array(words))


def test_raw_data_mutation(mock_metrics: RegressionMetricResult) -> None:
    """Verify raw_data dictionary can be mutated dynamically."""
    result = ConcreteResult(
        coefficients=np.array([0]), metrics=mock_metrics, raw_data={"key": "value"}
    )
    result.raw_data["new"] = "data"
    assert result.raw_data["new"] == "data"


@patch("xwhy.core.result.shap.Explanation")
def test_to_shap_conversion_success(
    mock_shap_explanation: MagicMock, mock_metrics: RegressionMetricResult
) -> None:
    """Verify to_shap creates an Explanation object correctly when shap is available."""
    result = TextXWhyResult(
        coefficients=np.array([0.5, 0.2]),
        metrics=mock_metrics,
        words=["SHAP", "test"],
    )

    out_object = result.to_shap()

    # 1. Verify it was called exactly once
    mock_shap_explanation.assert_called_once()

    # 2. Extract the arguments it was called with
    called_kwargs = mock_shap_explanation.call_args.kwargs

    # 3. Assert Numpy arrays using np.testing
    np.testing.assert_array_equal(called_kwargs["values"], result.coefficients)
    np.testing.assert_array_equal(called_kwargs["data"], result.data)

    # 4. Assert standard types normally
    assert called_kwargs["base_values"] == result.base_values
    assert list(called_kwargs["feature_names"]) == list(result.feature_names)

    # 5. Verify the returned object
    assert out_object == mock_shap_explanation.return_value


class TestBaseXWhyResult:
    """Test suite for the BaseXWhyResult class functionality."""

    def test_plot_raises_key_error_on_missing_data(
        self, mock_metrics: RegressionMetricResult
    ) -> None:
        """Ensure KeyError is raised if required arrays are missing in raw_data."""
        # Missing 'weights' and 'y_pred'
        raw_data = {"y_target": np.array([1.0, 2.0])}

        result_obj = ConcreteResult(
            coefficients=np.array([0.5, -0.5]),
            metrics=mock_metrics,
            raw_data=raw_data,
        )

        with pytest.raises(KeyError, match="'y_pred' must be present"):
            result_obj.plot()

    @patch("xwhy.core.result.plot_fidelity")
    def test_plot_success(
        self, mock_plot_fidelity: MagicMock, mock_metrics: RegressionMetricResult
    ) -> None:
        """Test that plot successfully delegates to plot_fidelity with correct data."""
        y_target_mock = np.array([1.0, 2.0])
        y_pred_mock = np.array([1.1, 1.9])
        weights_mock = np.array([1.0, 1.0])

        raw_data = {
            "y_target": y_target_mock,
            "y_pred": y_pred_mock,
            "weights": weights_mock,
            "extra_info": "should be ignored",
        }

        result_obj = ConcreteResult(
            coefficients=np.array([0.5, -0.5]),
            metrics=mock_metrics,
            raw_data=raw_data,
        )

        mock_plot_fidelity.return_value = "/mock/path/plot.png"

        save_path = Path("/mock/path/plot.png")
        returned_path = result_obj.plot(save_path=save_path, show=False)

        assert returned_path == "/mock/path/plot.png"

        # Verify the delegation was done correctly
        mock_plot_fidelity.assert_called_once_with(
            metrics=mock_metrics,
            y_target=y_target_mock,
            y_pred=y_pred_mock,
            weights=weights_mock,
            save_path=save_path,
            show=False,
        )
