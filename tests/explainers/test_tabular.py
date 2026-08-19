"""Unit tests for tabular explainer implementation."""

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xwhy.core.config import TabularConfig
from xwhy.explainers.tabular import TabularExplainer
from xwhy.models.tabular.adapter import TabularModelAdapter
from xwhy.surrogate.types import SurrogateType


@pytest.fixture
def mock_model() -> MagicMock:
    """Provide a mock model fixture with a predict method."""
    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 0, 1, 0])
    return model


def test_tabular_explainer_init_invalid_mode(mock_model: MagicMock) -> None:
    """Ensure ValueError is raised when mode is invalid."""
    with pytest.raises(
        ValueError, match=re.escape("mode must be 'classification' or 'regression'.")
    ):
        TabularExplainer(model=mock_model, mode="invalid_mode")


def test_tabular_explainer_init_with_custom_config(
    mock_model: MagicMock,
) -> None:
    """Verify initialization when a custom TabularConfig is provided."""
    config = TabularConfig(mode="regression", num_perturbations=10)
    explainer = TabularExplainer(model=mock_model, config=config)

    assert explainer.config == config
    assert isinstance(explainer.state.model, TabularModelAdapter)
    assert explainer.state.model.model == mock_model


@patch("xwhy.explainers.tabular.logger")
@patch("xwhy.explainers.tabular.calculate_distance")
@patch("xwhy.explainers.tabular.SurrogateTrainer")
@patch("xwhy.explainers.tabular.SurrogateFactory")
@patch("xwhy.explainers.tabular.RegressionMetrics")
@patch("xwhy.explainers.tabular.TabularXWhyResult")
def test_tabular_explainer_explain_classification_and_best_surrogate(
    mock_result_cls: MagicMock,
    mock_metrics_cls: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_calc_dist: MagicMock,
    mock_logger: MagicMock,
    mock_model: MagicMock,
) -> None:
    """Verify explain execution in classification mode with surrogate search.

    Tests normalization warning, bincount prediction aggregation, find_best
    surrogate search, classification y_pred thresholding, fidelity plot,
    and non-linear surrogate warnings.
    """
    mock_calc_dist.return_value = 0.1
    mock_trainer.find_best.return_value = (SurrogateType.LIME, 0.95)
    mock_trainer.compute_weights.return_value = np.array([1.0, 1.0])

    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.1, 0.2])
    mock_surrogate.predict.return_value = np.array([0.3, 0.7])
    mock_factory.create.return_value = mock_surrogate

    explainer = TabularExplainer(
        model=mock_model,
        mode="classification",
        num_perturbations=2,
        num_distribution_samples=5,
        use_best_surrogate=True,
        validate_normalization=True,
    )

    unnormalized_instance = np.array([10.0, 10.0])
    feature_names = ["feat1", "feat2"]

    result = explainer.explain(
        instance=unnormalized_instance,
        feature_names=feature_names,
        fidelity_plot=True,
    )

    # Verify both warnings were triggered (Surrogate & Unnormalized)
    assert mock_logger.warning.call_count == 2

    mock_logger.warning.assert_any_call(
        "Using a non-linear surrogate model or enabling 'use_best_surrogate' "
        "can replace a black-box model with another complex model, "
        "sacrificing local interpretability. The scientific community highly "
        "recommends utilizing simple linear models (e.g., LIME, OLS) to guarantee "
        "transparent and additive feature attributions."
    )

    mock_logger.warning.assert_any_call(
        "Instance appears not normalized. Ensure you pass standardized data."
    )

    # Verify surrogate search and plot triggering
    mock_trainer.find_best.assert_called_once()
    mock_result_cls.assert_called_once()
    mock_result_cls.return_value.plot.assert_called_once_with(show=True)
    assert result == mock_result_cls.return_value


@patch("xwhy.explainers.tabular.logger")
def test_tabular_explainer_nonlinear_surrogate_warning(
    mock_logger: MagicMock,
    mock_model: MagicMock,
) -> None:
    """Verify warning is logged when explicitly using a non-linear surrogate."""
    _ = TabularExplainer(
        model=mock_model,
        mode="classification",
        surrogate_type=SurrogateType.RANDOMFOREST,
        use_best_surrogate=False,
    )

    mock_logger.warning.assert_called_once_with(
        "Using a non-linear surrogate model or enabling 'use_best_surrogate' "
        "can replace a black-box model with another complex model, "
        "sacrificing local interpretability. The scientific community highly "
        "recommends utilizing simple linear models (e.g., LIME, OLS) to guarantee "
        "transparent and additive feature attributions."
    )


@patch("xwhy.explainers.tabular.logger")
@patch("xwhy.explainers.tabular.calculate_distance")
@patch("xwhy.explainers.tabular.SurrogateTrainer")
@patch("xwhy.explainers.tabular.SurrogateFactory")
@patch("xwhy.explainers.tabular.RegressionMetrics")
@patch("xwhy.explainers.tabular.TabularXWhyResult")
def test_tabular_explainer_explain_regression_and_default_surrogate(
    mock_result_cls: MagicMock,
    mock_metrics_cls: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_calc_dist: MagicMock,
    mock_logger: MagicMock,
    mock_model: MagicMock,
) -> None:
    """Verify explain execution in regression mode with default surrogate.

    Tests normalized instance check, regression prediction aggregation, skipping
    find_best search, regression y_pred flattening, and missing feature names.
    """
    mock_model.predict.return_value = np.array([1.5, 2.5])
    mock_calc_dist.return_value = 0.2
    mock_trainer.compute_weights.return_value = np.array([1.0, 1.0])

    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.5, -0.5])
    mock_surrogate.predict.return_value = np.array([1.2, 2.4])
    mock_factory.create.return_value = mock_surrogate

    explainer = TabularExplainer(
        model=mock_model,
        mode="regression",
        num_perturbations=2,
        num_distribution_samples=5,
        use_best_surrogate=False,
        validate_normalization=True,
    )

    normalized_instance = np.array([1.0, 2.0])

    result = explainer.explain(
        instance=normalized_instance,
        feature_names=None,
        fidelity_plot=False,
    )

    # Verify warning and surrogate search are skipped
    mock_logger.warning.assert_not_called()
    mock_trainer.find_best.assert_not_called()

    # Verify result creation and plot not rendered
    mock_result_cls.assert_called_once()
    mock_result_cls.return_value.plot.assert_not_called()
    assert result == mock_result_cls.return_value


def test_tabular_explainer_run_invalid_string_instance(mock_model: MagicMock) -> None:
    """Ensure TypeError is raised when a string is passed to run."""
    explainer = TabularExplainer(model=mock_model)
    with pytest.raises(TypeError, match="requires an array-like instance"):
        # Explicitly ignoring mypy error to test runtime validation
        explainer.run(instance="this is a string")


def test_tabular_explainer_run_invalid_numeric_instance(mock_model: MagicMock) -> None:
    """Ensure TypeError is raised when a numeric type is passed to run."""
    explainer = TabularExplainer(model=mock_model)
    with pytest.raises(TypeError, match="requires an array-like instance"):
        explainer.run(instance=12345)  # type: ignore[arg-type]


@patch.object(TabularExplainer, "explain")
def test_tabular_explainer_run_valid_delegation(
    mock_explain: MagicMock, mock_model: MagicMock
) -> None:
    """Verify run correctly delegates to explain with valid input and kwargs."""
    explainer = TabularExplainer(model=mock_model)
    valid_instance = np.array([1.5, 2.5, 3.5])
    mock_explain.return_value = MagicMock()

    # Call run with valid instance and additional kwargs
    result = explainer.run(
        instance=valid_instance,
        feature_names=["f1", "f2", "f3"],
        fidelity_plot=True,
        custom_kwarg="test",
    )

    # Assert explain was called exactly once with identical parameters
    mock_explain.assert_called_once_with(
        instance=valid_instance,
        feature_names=["f1", "f2", "f3"],
        fidelity_plot=True,
        custom_kwarg="test",
    )

    # Assert the return value matches what explain returned
    assert result == mock_explain.return_value


@patch("xwhy.explainers.tabular.SurrogateTrainer")
@patch("xwhy.explainers.tabular.SurrogateFactory")
@patch("xwhy.explainers.tabular.RegressionMetrics")
@patch("xwhy.explainers.tabular.calculate_distance")
def test_tabular_explain_impute_when_some_distances_valid(
    mock_calc_dist: MagicMock,
    mock_metrics: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
) -> None:
    """Cover the branch where at least one scaled distance is finite."""
    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 0])

    explainer = TabularExplainer(
        model=model,
        num_perturbations=3,
        num_distribution_samples=5,
        use_best_surrogate=False,
        seed=42,
        validate_normalization=False,
    )

    # Force some non-finite distances inside the loop
    # by making calculate_distance return mixed values
    mock_calc_dist.side_effect = [0.5, 0.5, np.inf, 0.5, 1.5, 1.5] * 10

    mock_trainer.compute_weights.return_value = np.ones(3)
    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.1, 0.2])
    mock_surrogate.predict.return_value = np.array([0.5, 0.6, 0.7])
    mock_factory.create.return_value = mock_surrogate
    mock_metrics.calculate.return_value = MagicMock()

    instance = np.array([0.1, -0.2])
    result = explainer.explain(instance)

    distances = result.raw_data["distances"]
    # At least one entry must have been imputed with max+1000
    assert np.any(distances > 1000.0)
    assert np.all(np.isfinite(distances))


@patch("xwhy.explainers.tabular.SurrogateTrainer")
@patch("xwhy.explainers.tabular.SurrogateFactory")
@patch("xwhy.explainers.tabular.RegressionMetrics")
@patch("xwhy.explainers.tabular.calculate_distance")
def test_tabular_explain_impute_when_all_distances_non_finite(
    mock_calc_dist: MagicMock,
    mock_metrics: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
) -> None:
    """Cover the branch where every scaled distance is non-finite."""
    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 0])

    explainer = TabularExplainer(
        model=model,
        num_perturbations=2,
        num_distribution_samples=3,
        use_best_surrogate=False,
        seed=42,
        validate_normalization=False,
    )

    mock_calc_dist.return_value = np.inf

    mock_trainer.compute_weights.return_value = np.ones(2)
    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.1, 0.2])
    mock_surrogate.predict.return_value = np.array([0.5, 0.5])
    mock_factory.create.return_value = mock_surrogate
    mock_metrics.calculate.return_value = MagicMock()

    instance = np.array([0.0, 0.0])
    result = explainer.explain(instance)

    distances = result.raw_data["distances"]
    assert np.allclose(distances, 1000.0)
