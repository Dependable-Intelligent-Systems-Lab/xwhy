"""Unit tests for core results."""

from collections.abc import Sequence
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xwhy.core.result import (
    BaseXWhyResult,
    ImageClassificationXWhyResult,
    ImageGenerationAndEditingXWhyResult,
    TabularXWhyResult,
    TextXWhyResult,
)
from xwhy.metrics.regression import RegressionMetricResult


class ConcreteResult(BaseXWhyResult):
    """Concrete implementation of BaseXWhyResult for testing."""

    @property
    def feature_names(self) -> Sequence[str]:
        """Return mock feature names for testing."""
        return ["feat1", "feat2"]

    @property
    def data(self) -> np.ndarray:
        """Return mock data instance for testing."""
        return np.array([1, 2])


@pytest.fixture
def mock_metrics() -> RegressionMetricResult:
    """Provide a dummy metric result fixture.

    Returns:
        RegressionMetricResult: A mocked metric result object.

    """
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


def test_text_result_word_importances(mock_metrics: RegressionMetricResult) -> None:
    """Verify that word_importances correctly zips words with float coefficients."""
    coeffs = np.array([0.5, -1.2, 3.0])
    words = ["word1", "word2", "word3"]
    result = TextXWhyResult(
        coefficients=coeffs,
        metrics=mock_metrics,
        original_output="output",
        words=words,
    )

    importances = result.word_importances

    assert isinstance(importances, list)
    assert len(importances) == 3
    assert importances == [
        ("word1", 0.5),
        ("word2", -1.2),
        ("word3", 3.0),
    ]


def test_raw_data_mutation(mock_metrics: RegressionMetricResult) -> None:
    """Verify raw_data dictionary can be mutated dynamically."""
    result = ConcreteResult(
        coefficients=np.array([0]), metrics=mock_metrics, raw_data={"key": "value"}
    )
    result.raw_data["new"] = "data"
    assert result.raw_data["new"] == "data"


@patch("xwhy.core.result.Explanation")
def test_to_shap_conversion_success(
    mock_shap_explanation: MagicMock, mock_metrics: RegressionMetricResult
) -> None:
    """Verify to_shap creates an Explanation object correctly."""
    result = TextXWhyResult(
        coefficients=np.array([0.5, 0.2]),
        metrics=mock_metrics,
        words=["SHAP", "test"],
    )

    out_object = result.to_shap()

    mock_shap_explanation.assert_called_once()
    called_kwargs = mock_shap_explanation.call_args.kwargs

    np.testing.assert_array_equal(called_kwargs["values"], result.coefficients)
    np.testing.assert_array_equal(called_kwargs["data"], result.data)

    assert called_kwargs["base_values"] == result.base_values
    assert list(called_kwargs["feature_names"]) == list(result.feature_names)
    assert out_object == mock_shap_explanation.return_value


class TestBaseXWhyResult:
    """Test suite for the BaseXWhyResult class functionality."""

    def test_plot_raises_key_error_on_missing_data(
        self, mock_metrics: RegressionMetricResult
    ) -> None:
        """Ensure KeyError is raised if required arrays are missing in raw_data."""
        raw_data = {"y_target": np.array([1.0, 2.0])}

        result_obj = ConcreteResult(
            coefficients=np.array([0.5, -0.5]),
            metrics=mock_metrics,
            raw_data=raw_data,
        )

        with pytest.raises(KeyError, match="'y_pred' must be present"):
            result_obj.plot()

    @patch("xwhy.plots.metrics.plot_fidelity")
    def test_plot_success(
        self, mock_plot_fidelity: MagicMock, mock_metrics: RegressionMetricResult
    ) -> None:
        """Test that plot successfully delegates to plot_fidelity."""
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

        mock_plot_fidelity.assert_called_once_with(
            metrics=mock_metrics,
            y_target=y_target_mock,
            y_pred=y_pred_mock,
            weights=weights_mock,
            save_path=save_path,
            show=False,
        )


def test_image_classification_result_initialization(
    mock_metrics: RegressionMetricResult,
) -> None:
    """Verify ImageClassificationXWhyResult initialization with defaults."""
    coeffs = np.array([0.1, 0.2])
    orig_img = np.zeros((10, 10, 3))
    superpixels = np.ones((10, 10))
    top_features = np.array([0, 1])

    result = ImageClassificationXWhyResult(
        coefficients=coeffs,
        metrics=mock_metrics,
        original_image=orig_img,
        superpixels=superpixels,
        top_features=top_features,
        coverage=0.8,
        weighted_coverage=0.75,
    )

    np.testing.assert_array_equal(result.coefficients, coeffs)
    assert result.metrics == mock_metrics
    np.testing.assert_array_equal(result.original_image, orig_img)
    np.testing.assert_array_equal(result.superpixels, superpixels)
    np.testing.assert_array_equal(result.top_features, top_features)
    assert result.coverage == 0.8
    assert result.weighted_coverage == 0.75


def test_image_classification_result_properties(
    mock_metrics: RegressionMetricResult,
) -> None:
    """Verify feature_names and data properties of image results."""
    coeffs = np.array([0.5, 0.3])
    orig_img = np.ones((5, 5, 3))
    result = ImageClassificationXWhyResult(
        coefficients=coeffs,
        metrics=mock_metrics,
        original_image=orig_img,
    )

    assert result.feature_names == ["Superpixel 0", "Superpixel 1"]
    np.testing.assert_array_equal(result.data, orig_img)


@patch("xwhy.core.result.Explanation")
@patch("xwhy.plots.image.create_image_heat_mask")
def test_to_shap_with_superpixels_ndim_3(
    mock_create_mask: MagicMock,
    mock_shap_explanation: MagicMock,
    mock_metrics: RegressionMetricResult,
) -> None:
    """Verify to_shap with superpixels for 3D image."""
    coeffs = np.array([0.5])
    orig_img = np.zeros((10, 10, 3))
    superpixels = np.ones((10, 10), dtype=int)
    mock_create_mask.return_value = np.zeros((10, 10))

    result = ImageClassificationXWhyResult(
        coefficients=coeffs,
        metrics=mock_metrics,
        original_image=orig_img,
        superpixels=superpixels,
    )

    out_obj = result.to_shap()

    mock_create_mask.assert_called_once_with(superpixels, coeffs)
    mock_shap_explanation.assert_called_once()
    assert out_obj == mock_shap_explanation.return_value


@patch("xwhy.core.result.Explanation")
@patch("xwhy.plots.image.create_image_heat_mask")
def test_to_shap_with_superpixels_ndim_2(
    mock_create_mask: MagicMock,
    mock_shap_explanation: MagicMock,
    mock_metrics: RegressionMetricResult,
) -> None:
    """Verify to_shap with superpixels for 2D image."""
    coeffs = np.array([0.5])
    orig_img = np.zeros((10, 10))
    superpixels = np.ones((10, 10), dtype=int)
    mock_create_mask.return_value = np.zeros((10, 10))

    result = ImageClassificationXWhyResult(
        coefficients=coeffs,
        metrics=mock_metrics,
        original_image=orig_img,
        superpixels=superpixels,
    )

    out_obj = result.to_shap()

    mock_create_mask.assert_called_once()
    mock_shap_explanation.assert_called_once()
    assert out_obj == mock_shap_explanation.return_value


@patch("xwhy.core.result.Explanation")
@patch("xwhy.plots.image.create_image_heat_mask")
def test_to_shap_with_superpixels_ndim_4(
    mock_create_mask: MagicMock,
    mock_shap_explanation: MagicMock,
    mock_metrics: RegressionMetricResult,
) -> None:
    """Verify to_shap with superpixels for 4D image."""
    coeffs = np.array([0.5])
    orig_img = np.zeros((1, 10, 10, 3))
    superpixels = np.ones((10, 10), dtype=int)
    mock_create_mask.return_value = np.zeros((10, 10))

    result = ImageClassificationXWhyResult(
        coefficients=coeffs,
        metrics=mock_metrics,
        original_image=orig_img,
        superpixels=superpixels,
    )

    out_obj = result.to_shap()

    mock_create_mask.assert_called_once()
    mock_shap_explanation.assert_called_once()
    assert out_obj == mock_shap_explanation.return_value


@patch("xwhy.core.result.Explanation")
def test_to_shap_without_superpixels(
    mock_shap_explanation: MagicMock,
    mock_metrics: RegressionMetricResult,
) -> None:
    """Verify to_shap conversion with empty superpixels."""
    coeffs = np.array([0.5, 0.2])
    orig_img = np.zeros((10, 10, 3))

    result = ImageClassificationXWhyResult(
        coefficients=coeffs,
        metrics=mock_metrics,
        original_image=orig_img,
        superpixels=np.zeros(0),
    )

    out_obj = result.to_shap()

    mock_shap_explanation.assert_called_once()
    called_kwargs = mock_shap_explanation.call_args.kwargs
    np.testing.assert_array_equal(called_kwargs["values"], coeffs)
    np.testing.assert_array_equal(called_kwargs["data"], orig_img)
    assert out_obj == mock_shap_explanation.return_value


def test_tabular_result_initialization(mock_metrics: RegressionMetricResult) -> None:
    """Verify TabularXWhyResult initializes correctly and maps the data property."""
    coeffs = np.array([0.1, 0.2])
    instance = np.array([10.5, 20.1])
    result = TabularXWhyResult(
        coefficients=coeffs,
        metrics=mock_metrics,
        instance=instance,
    )

    assert np.array_equal(result.coefficients, coeffs)
    assert result.metrics == mock_metrics
    assert result.data is not None
    assert np.array_equal(result.data, instance)


def test_tabular_feature_names_explicit(mock_metrics: RegressionMetricResult) -> None:
    """Verify explicit feature list is returned when provided."""
    features = ["Age", "Income"]
    result = TabularXWhyResult(
        coefficients=np.array([0.1, 0.2]),
        metrics=mock_metrics,
        feature_list=features,
        instance=np.array([25, 50000]),
    )

    assert result.feature_names == features


def test_tabular_feature_names_generated(mock_metrics: RegressionMetricResult) -> None:
    """Verify feature names are dynamically generated when list is missing."""
    result = TabularXWhyResult(
        coefficients=np.array([0.1, 0.2]),
        metrics=mock_metrics,
        feature_list=[],
        instance=np.array([25, 50000]),
    )

    expected_names = ["Feature_0", "Feature_1"]
    assert result.feature_names == expected_names


def test_tabular_feature_names_empty(mock_metrics: RegressionMetricResult) -> None:
    """Verify empty list is returned when both feature_list and instance are missing."""
    result = TabularXWhyResult(
        coefficients=np.array([0.1, 0.2]),
        metrics=mock_metrics,
        feature_list=[],
        instance=None,
    )

    assert result.feature_names == []


def test_image_result_with_words(mock_metrics: RegressionMetricResult) -> None:
    """Test result properties when words are provided."""
    result = ImageGenerationAndEditingXWhyResult(
        coefficients=np.array([0.1, 0.2]),
        metrics=mock_metrics,
        words=["hello", "world"],
    )
    assert list(result.feature_names) == ["hello", "world"]
    np.testing.assert_array_equal(result.data, np.array(["hello", "world"]))


def test_image_result_with_string_instance(
    mock_metrics: RegressionMetricResult,
) -> None:
    """Test result data property when instance is a string."""
    result = ImageGenerationAndEditingXWhyResult(
        coefficients=np.array([0.1, 0.2]),
        metrics=mock_metrics,
        words=[],
        instance="test prompt string",
    )
    assert list(result.feature_names) == []
    np.testing.assert_array_equal(result.data, np.array(["test", "prompt", "string"]))


def test_image_result_with_ndarray_instance(
    mock_metrics: RegressionMetricResult,
) -> None:
    """Test result data property when instance is a numpy array."""
    arr = np.array([1, 2, 3])
    result = ImageGenerationAndEditingXWhyResult(
        coefficients=np.array([0.1, 0.2]), metrics=mock_metrics, words=[], instance=arr
    )
    assert list(result.feature_names) == []
    np.testing.assert_array_equal(result.data, np.array([str(arr)]))


def test_image_result_with_none_instance(mock_metrics: RegressionMetricResult) -> None:
    """Test result data property when instance is None and words are empty."""
    result = ImageGenerationAndEditingXWhyResult(
        coefficients=np.array([0.1, 0.2]),
        metrics=mock_metrics,
        words=[],
        instance=None,
    )
    assert list(result.feature_names) == []
    np.testing.assert_array_equal(result.data, np.array([]))
