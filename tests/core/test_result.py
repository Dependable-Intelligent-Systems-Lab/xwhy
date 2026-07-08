"""Unit tests for core results."""

import sys
from collections.abc import Sequence
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


def test_to_shap_conversion_success(mock_metrics: RegressionMetricResult) -> None:
    """Verify to_shap creates an Explanation object correctly when shap is available."""
    mock_shap_explanation = MagicMock()
    mock_shap_module = MagicMock()
    mock_shap_module.Explanation = mock_shap_explanation

    result = TextXWhyResult(
        coefficients=np.array([0.5, 0.2]),
        metrics=mock_metrics,
        words=["SHAP", "test"],
    )

    with patch.dict(sys.modules, {"shap": mock_shap_module}):
        out_object = result.to_shap()

    # 1. Verify it was called exactly once
    assert mock_shap_explanation.call_count == 1

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


def test_to_shap_conversion_missing_library(
    mock_metrics: RegressionMetricResult,
) -> None:
    """Ensure an explicit ImportError is raised if SHAP is absent."""
    result = TextXWhyResult(
        coefficients=np.array([0.5]),
        metrics=mock_metrics,
        words=["missing"],
    )

    with (
        patch.dict(sys.modules, {"shap": None}),
        pytest.raises(ImportError, match="The 'shap' library is required"),
    ):
        result.to_shap()
