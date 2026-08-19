"""Unit tests for the TextExplainer class."""

import re
from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xwhy.core.result import TextXWhyResult
from xwhy.explainers.text import TextExplainer
from xwhy.models.embeddings.types import EmbeddingType
from xwhy.surrogate.types import SurrogateType


class MockModelProba:
    """Mock model implementing predict_proba."""

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Mock probabilistic predictions.

        Args:
            texts: Input sequence of text strings.

        Returns:
            np.ndarray: Mocked 2D array of probabilities.

        """
        return np.array([[0.2, 0.8] for _ in texts])


class MockModelPredict:
    """Mock model implementing predict."""

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Mock binary/class predictions.

        Args:
            texts: Input sequence of text strings.

        Returns:
            np.ndarray: Mocked 1D array of class indices.

        """
        return np.array([1 for _ in texts])


class MockModelCallable:
    """Mock model implementing __call__."""

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        """Mock callable predictions.

        Args:
            texts: Input sequence of text strings.

        Returns:
            np.ndarray: Mocked 1D array.

        """
        return np.array([1 for _ in texts])


class MockModelInvalid:
    """Mock model lacking any prediction methods."""


def dummy_predict_fn(texts: Sequence[str]) -> np.ndarray:
    """Mock standalone prediction function.

    Args:
        texts: Input sequence of text strings.

    Returns:
        np.ndarray: Mocked array of predictions.

    """
    return np.array([1 for _ in texts])


@pytest.fixture
def base_explainer() -> TextExplainer:
    """Fixture providing a baseline TextExplainer with mocked internals.

    Returns:
        TextExplainer: Instantiated explainer with mocked factories.

    """
    with (
        patch("xwhy.explainers.text.TextConfig") as mock_config_cls,
        patch("xwhy.explainers.text.EmbeddingFactory"),
        patch("xwhy.explainers.text.TextPerturbation"),
    ):
        mock_config = MagicMock()
        mock_config.embedding_type = EmbeddingType.WORD2VEC
        mock_config.surrogate_type = SurrogateType.LIME
        mock_config.seed = 42
        mock_config.num_perturbations = 64
        mock_config.use_best_surrogate = True
        mock_config_cls.return_value = mock_config

        return TextExplainer(predict_fn=dummy_predict_fn)


def test_init_invalid_embedding_type() -> None:
    """Test initialization fails when embedding type is not a text embedding."""
    with (
        patch("xwhy.explainers.text.EmbeddingType") as mock_embedding_type,
        patch("xwhy.explainers.text.TextConfig"),
    ):
        mock_invalid = MagicMock()
        mock_invalid.is_text_embedding = False
        mock_embedding_type.from_str.return_value = mock_invalid

        with pytest.raises(ValueError, match="Must be a text embedding"):
            TextExplainer(
                predict_fn=dummy_predict_fn,
                embedding_type="invalid_type",
            )


def test_init_logger_warning_for_non_linear_surrogate() -> None:
    """Test a warning is logged when a non-linear surrogate configuration is used."""
    with (
        patch("xwhy.explainers.text.logger") as mock_logger,
        patch("xwhy.explainers.text.TextConfig") as mock_config_cls,
        patch("xwhy.explainers.text.EmbeddingFactory"),
        patch("xwhy.explainers.text.TextPerturbation"),
    ):
        mock_config = MagicMock()
        mock_config.embedding_type = EmbeddingType.WORD2VEC
        mock_config.surrogate_type = SurrogateType.LIME
        mock_config.seed = 42
        mock_config.num_perturbations = 64
        mock_config.use_best_surrogate = True
        mock_config_cls.return_value = mock_config

        TextExplainer(predict_fn=dummy_predict_fn, use_best_surrogate=True)
        mock_logger.warning.assert_called_once()


def test_init_warning_for_non_linear_surrogate_type() -> None:
    """Test a warning is logged for non-linear models even if best search is off."""
    with (
        patch("xwhy.explainers.text.logger") as mock_logger,
        patch("xwhy.explainers.text.TextConfig") as mock_config_cls,
        patch("xwhy.explainers.text.EmbeddingFactory"),
        patch("xwhy.explainers.text.TextPerturbation"),
    ):
        mock_config = MagicMock()
        mock_config.embedding_type = EmbeddingType.WORD2VEC
        mock_config.use_best_surrogate = False

        mock_surrogate_type = MagicMock()
        mock_surrogate_type.is_linear_model = False
        mock_config.surrogate_type = mock_surrogate_type

        mock_config_cls.return_value = mock_config

        TextExplainer(predict_fn=dummy_predict_fn, use_best_surrogate=False)
        mock_logger.warning.assert_called_once()


def test_init_no_warning_for_linear_surrogate() -> None:
    """Test no warning is logged if use_best_surrogate is False and model is linear."""
    with (
        patch("xwhy.explainers.text.logger") as mock_logger,
        patch("xwhy.explainers.text.TextConfig") as mock_config_cls,
        patch("xwhy.explainers.text.EmbeddingFactory"),
        patch("xwhy.explainers.text.TextPerturbation"),
    ):
        mock_config = MagicMock()
        mock_config.embedding_type = EmbeddingType.WORD2VEC
        mock_config.use_best_surrogate = False

        mock_surrogate_type = MagicMock()
        mock_surrogate_type.is_linear_model = True
        mock_config.surrogate_type = mock_surrogate_type

        mock_config_cls.return_value = mock_config

        TextExplainer(predict_fn=dummy_predict_fn, use_best_surrogate=False)
        mock_logger.warning.assert_not_called()


def test_init_creates_default_config() -> None:
    """Test passing parameters directly to __init__ instantiates a TextConfig."""
    with (
        patch("xwhy.explainers.text.TextConfig") as mock_config_cls,
        patch("xwhy.explainers.text.EmbeddingFactory"),
        patch("xwhy.explainers.text.TextPerturbation"),
    ):
        mock_config = MagicMock()
        mock_config.embedding_type = EmbeddingType.WORD2VEC
        mock_config.surrogate_type = SurrogateType.LIME
        mock_config.use_best_surrogate = False
        mock_config_cls.return_value = mock_config

        explainer = TextExplainer(
            predict_fn=dummy_predict_fn,
            seed=123,
            num_perturbations=32,
            embedding_type="word2vec",
            surrogate_type="lime_ridge",
            use_best_surrogate=False,
        )

        mock_config_cls.assert_called_once_with(
            model=None,
            predict_fn=dummy_predict_fn,
            seed=123,
            num_perturbations=32,
            embedding_type="word2vec",
            surrogate_type="lime_ridge",
            use_best_surrogate=False,
        )
        assert explainer.config is mock_config


def test_init_with_explicit_config() -> None:
    """Test initialization bypasses TextConfig creation when a config is provided."""
    with (
        patch("xwhy.explainers.text.TextConfig") as mock_config_cls,
        patch("xwhy.explainers.text.EmbeddingFactory"),
        patch("xwhy.explainers.text.TextPerturbation"),
    ):
        mock_config = MagicMock()
        mock_config.embedding_type = EmbeddingType.WORD2VEC
        mock_config.use_best_surrogate = False
        mock_config.surrogate_type.is_linear_model = True
        mock_config.model = None
        mock_config.predict_fn = dummy_predict_fn

        explainer = TextExplainer(config=mock_config)

        mock_config_cls.assert_not_called()
        assert explainer.config is mock_config


def test_initialize_without_model_or_predict_fn() -> None:
    """Test _initialize does not overwrite state if both model and predict_fn absent."""
    with (
        patch("xwhy.explainers.text.TextConfig") as mock_config_cls,
        patch("xwhy.explainers.text.EmbeddingFactory"),
        patch("xwhy.explainers.text.TextPerturbation"),
    ):
        mock_config = MagicMock()
        mock_config.embedding_type = EmbeddingType.WORD2VEC
        mock_config.use_best_surrogate = False
        mock_config.surrogate_type.is_linear_model = True
        mock_config.model = None
        mock_config.predict_fn = None
        mock_config_cls.return_value = mock_config

        explainer = TextExplainer(config=mock_config)
        assert getattr(explainer.state, "predict_fn", None) is None


def test_resolve_predict_fn_callable() -> None:
    """Test resolving a direct callable prediction function."""
    resolved = TextExplainer._resolve_predict_fn(predict_fn=dummy_predict_fn)
    assert resolved is dummy_predict_fn


def test_resolve_predict_fn_not_callable() -> None:
    """Test resolving fails when predict_fn is not actually callable."""
    with pytest.raises(
        TypeError, match=re.escape("Provided 'predict_fn' must be callable.")
    ):
        TextExplainer._resolve_predict_fn(predict_fn="not_callable")  # type: ignore[arg-type]


def test_resolve_predict_fn_model_predict_proba() -> None:
    """Test resolving a model utilizing predict_proba."""
    model = MockModelProba()
    resolved = TextExplainer._resolve_predict_fn(model=model)
    assert resolved.__name__ == "predict_proba"


def test_resolve_predict_fn_model_predict() -> None:
    """Test resolving a model utilizing predict."""
    model = MockModelPredict()
    resolved = TextExplainer._resolve_predict_fn(model=model)
    assert resolved.__name__ == "predict"


def test_resolve_predict_fn_model_callable() -> None:
    """Test resolving a model utilizing __call__."""
    model = MockModelCallable()
    resolved = TextExplainer._resolve_predict_fn(model=model)
    assert resolved is model


def test_resolve_predict_fn_model_invalid() -> None:
    """Test resolving fails when model lacks supported prediction methods."""
    model = MockModelInvalid()
    with pytest.raises(ValueError, match="Provided model must be callable"):
        TextExplainer._resolve_predict_fn(model=model)


def test_resolve_predict_fn_neither_provided() -> None:
    """Test resolving fails when neither model nor predict_fn is supplied."""
    with pytest.raises(ValueError, match="Either 'model' or 'predict_fn'"):
        TextExplainer._resolve_predict_fn()


def test_initialize_invalid_embedding_state(base_explainer: TextExplainer) -> None:
    """Test the safety check inside _initialize for invalid embedding types."""
    base_explainer.config.embedding_type = MagicMock(is_text_embedding=False)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="Must be a text embedding"):
        base_explainer._initialize()


def test_explain_requires_string_instance(base_explainer: TextExplainer) -> None:
    """Test explaining an instance that is not a string raises TypeError."""
    with pytest.raises(TypeError, match="requires the input text as a string"):
        base_explainer.explain(instance=123)  # type: ignore[arg-type]


def test_explain_no_predict_fn_available(base_explainer: TextExplainer) -> None:
    """Test explain fails if state lacks a predict_fn and none is provided."""
    base_explainer.state.predict_fn = None
    with pytest.raises(ValueError, match="No prediction model or predict_fn"):
        base_explainer.explain(instance="test")


def test_explain_perturbator_uninitialized(base_explainer: TextExplainer) -> None:
    """Test explain fails if the perturbator was never initialized."""
    base_explainer.state.perturbator = None
    with pytest.raises(RuntimeError, match="TextPerturbation state is not"):
        base_explainer.explain(instance="test")


def test_explain_embedding_uninitialized(base_explainer: TextExplainer) -> None:
    """Test explain fails if the embedding model was never initialized."""
    mock_pert = MagicMock()
    mock_pert.generate.return_value = (["test"], [[1]])
    base_explainer.state.perturbator = mock_pert

    base_explainer.state.embedding_model = None
    with pytest.raises(RuntimeError, match="Embedding model state is not"):
        base_explainer.explain(instance="test")


@patch("xwhy.explainers.text.WMDDistance")
@patch("xwhy.explainers.text.SurrogateTrainer")
@patch("xwhy.explainers.text.SurrogateFactory")
@patch("xwhy.explainers.text.RegressionMetrics")
def test_explain_with_explicit_predict_fn(
    mock_metrics: MagicMock,
    mock_surr_factory: MagicMock,
    mock_surr_trainer: MagicMock,
    mock_wmd: MagicMock,
    base_explainer: TextExplainer,
) -> None:
    """Test explain() overrides internal state when given an explicit predict_fn."""
    mock_pert = MagicMock()
    mock_pert.generate.return_value = (["test"], [[1]])
    base_explainer.state.perturbator = mock_pert

    mock_wmd_instance = mock_wmd.return_value
    mock_wmd_instance.compute_batch.return_value = [("test", 0.0)]

    mock_surr_trainer.find_best.return_value = (MagicMock(), 0.9)
    mock_surr_trainer.compute_weights.return_value = np.array([1.0])

    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.5])
    mock_surrogate.predict.return_value = np.array([1.0])
    mock_surr_factory.create.return_value = mock_surrogate

    mock_metrics.calculate.return_value = MagicMock()

    def custom_predict(texts: Sequence[str]) -> np.ndarray:
        return np.array([2.0])

    result = base_explainer.explain(
        instance="test",
        predict_fn=custom_predict,
        num_perturbations=10,
    )

    assert isinstance(result, TextXWhyResult)
    mock_surr_trainer.find_best.assert_called_once()


@patch("xwhy.explainers.text.WMDDistance")
@patch("xwhy.explainers.text.SurrogateTrainer")
@patch("xwhy.explainers.text.SurrogateFactory")
@patch("xwhy.explainers.text.RegressionMetrics")
def test_explain_1d_predictions_and_best_surrogate(
    mock_metrics: MagicMock,
    mock_surr_factory: MagicMock,
    mock_surr_trainer: MagicMock,
    mock_wmd: MagicMock,
    base_explainer: TextExplainer,
) -> None:
    """Test execution path handling 1D predictions and surrogate optimization."""
    mock_pert = MagicMock()
    mock_pert.generate.return_value = (["test"], [[1]])
    base_explainer.state.perturbator = mock_pert

    mock_wmd_instance = mock_wmd.return_value
    mock_wmd_instance.compute_batch.return_value = [("test", 0.0)]

    mock_surr_trainer.find_best.return_value = (MagicMock(), 0.9)
    mock_surr_trainer.compute_weights.return_value = np.array([1.0])

    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.5])
    mock_surrogate.predict.return_value = np.array([1.0])
    mock_surr_factory.create.return_value = mock_surrogate

    mock_metrics.calculate.return_value = MagicMock()

    base_explainer.config.use_best_surrogate = True  # type: ignore[union-attr]

    def mock_predict_1d(texts: Sequence[str]) -> np.ndarray:
        return np.array([1.0])

    # Assign to state explicitly instead of passing to explain() to hit internal branch
    base_explainer.state.predict_fn = mock_predict_1d

    result = base_explainer.explain(
        instance="test",
        num_perturbations=10,
    )

    assert isinstance(result, TextXWhyResult)
    mock_surr_trainer.find_best.assert_called_once()


@patch("xwhy.explainers.text.WMDDistance")
@patch("xwhy.explainers.text.SurrogateTrainer")
@patch("xwhy.explainers.text.SurrogateFactory")
@patch("xwhy.explainers.text.RegressionMetrics")
@patch.object(TextXWhyResult, "plot")
def test_explain_2d_predictions_default_surrogate_and_plot(
    mock_plot: MagicMock,
    mock_metrics: MagicMock,
    mock_surr_factory: MagicMock,
    mock_surr_trainer: MagicMock,
    mock_wmd: MagicMock,
    base_explainer: TextExplainer,
) -> None:
    """Test execution with 2D arrays, default surrogate use, and fidelity plot."""
    mock_pert = MagicMock()
    mock_pert.generate.return_value = (["test"], [[1]])
    base_explainer.state.perturbator = mock_pert

    mock_wmd_instance = mock_wmd.return_value
    mock_wmd_instance.compute_batch.return_value = [("test", 0.0)]

    mock_surr_trainer.compute_weights.return_value = np.array([1.0])

    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.5])
    mock_surrogate.predict.return_value = np.array([1.0])
    mock_surr_factory.create.return_value = mock_surrogate

    mock_metrics.calculate.return_value = MagicMock()

    base_explainer.config.use_best_surrogate = False  # type: ignore[union-attr]

    def mock_predict_2d(texts: Sequence[str]) -> np.ndarray:
        return np.array([[0.1, 0.9]])

    result = base_explainer.explain(
        instance="test",
        predict_fn=mock_predict_2d,
        fidelity_plot=True,
    )

    assert isinstance(result, TextXWhyResult)
    mock_surr_trainer.find_best.assert_not_called()
    mock_plot.assert_called_once_with(show=True)


@patch("xwhy.explainers.text.WMDDistance")
@patch("xwhy.explainers.text.SurrogateTrainer")
@patch("xwhy.explainers.text.SurrogateFactory")
@patch("xwhy.explainers.text.RegressionMetrics")
def test_explain_empty_predictions(
    mock_metrics: MagicMock,
    mock_surr_factory: MagicMock,
    mock_surr_trainer: MagicMock,
    mock_wmd: MagicMock,
    base_explainer: TextExplainer,
) -> None:
    """Test generation of explanation handles empty prediction edge cases."""
    mock_pert = MagicMock()
    mock_pert.generate.return_value = (["test"], [[1]])
    base_explainer.state.perturbator = mock_pert

    mock_wmd_instance = mock_wmd.return_value
    mock_wmd_instance.compute_batch.return_value = [("test", 0.0)]

    mock_surr_trainer.find_best.return_value = (MagicMock(), 0.0)
    mock_surr_trainer.compute_weights.return_value = np.array([])

    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([])
    mock_surrogate.predict.return_value = np.array([])
    mock_surr_factory.create.return_value = mock_surrogate
    mock_metrics.calculate.return_value = MagicMock()

    def mock_predict_empty(texts: Sequence[str]) -> np.ndarray:
        return np.array([])

    result = base_explainer.explain(
        instance="test",
        predict_fn=mock_predict_empty,
        class_index=0,
    )

    assert isinstance(result, TextXWhyResult)


@patch("xwhy.explainers.text.SurrogateTrainer")
@patch("xwhy.explainers.text.SurrogateFactory")
@patch("xwhy.explainers.text.RegressionMetrics")
@patch("xwhy.explainers.text.WMDDistance")
def test_text_explain_impute_when_some_distances_valid(
    mock_wmd: MagicMock,
    mock_metrics: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
) -> None:
    """Cover the branch where at least one WMD distance is finite."""
    predict_fn = MagicMock(return_value=np.array([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]))

    with (
        patch("xwhy.explainers.text.EmbeddingFactory"),
        patch("xwhy.explainers.text.TextPerturbation") as mock_pert,
    ):
        mock_pert.return_value.generate.return_value = (
            ["t1", "t2", "t3"],
            [[1, 0], [0, 1], [1, 1]],
        )
        explainer = TextExplainer(predict_fn=predict_fn, use_best_surrogate=False)

    mock_wmd.return_value.compute_batch.return_value = [
        ("t1", 0.5),
        ("t2", np.inf),
        ("t3", 1.5),
    ]
    mock_trainer.compute_weights.return_value = np.ones(3)
    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.1, 0.2])
    mock_surrogate.predict.return_value = np.array([0.5, 0.6, 0.7])
    mock_factory.create.return_value = mock_surrogate
    mock_metrics.calculate.return_value = MagicMock()

    result = explainer.explain("hello world")

    distances = result.raw_data["distances"]
    assert distances[0] == pytest.approx(0.5)
    assert distances[1] == pytest.approx(1001.5)
    assert distances[2] == pytest.approx(1.5)


@patch("xwhy.explainers.text.SurrogateTrainer")
@patch("xwhy.explainers.text.SurrogateFactory")
@patch("xwhy.explainers.text.RegressionMetrics")
@patch("xwhy.explainers.text.WMDDistance")
def test_text_explain_impute_when_all_distances_non_finite(
    mock_wmd: MagicMock,
    mock_metrics: MagicMock,
    mock_factory: MagicMock,
    mock_trainer: MagicMock,
) -> None:
    """Cover the branch where every WMD distance is non-finite."""
    predict_fn = MagicMock(return_value=np.array([[0.2, 0.8], [0.3, 0.7]]))

    with (
        patch("xwhy.explainers.text.EmbeddingFactory"),
        patch("xwhy.explainers.text.TextPerturbation") as mock_pert,
    ):
        mock_pert.return_value.generate.return_value = (
            ["t1", "t2"],
            [[1, 0], [0, 1]],
        )
        explainer = TextExplainer(predict_fn=predict_fn, use_best_surrogate=False)

    mock_wmd.return_value.compute_batch.return_value = [
        ("t1", np.inf),
        ("t2", np.nan),
    ]
    mock_trainer.compute_weights.return_value = np.ones(2)
    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.1])
    mock_surrogate.predict.return_value = np.array([0.5, 0.5])
    mock_factory.create.return_value = mock_surrogate
    mock_metrics.calculate.return_value = MagicMock()

    result = explainer.explain("hello world")

    distances = result.raw_data["distances"]
    assert distances[0] == pytest.approx(1000.0)
    assert distances[1] == pytest.approx(1000.0)
