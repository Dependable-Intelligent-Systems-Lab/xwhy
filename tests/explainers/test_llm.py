"""Tests for the LLM explainer module."""

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xwhy.core.config import LLMConfig
from xwhy.core.result import TextXWhyResult
from xwhy.explainers.llm import LLMExplainer
from xwhy.providers.base import BaseProvider
from xwhy.providers.types import ProviderType
from xwhy.surrogate.types import SurrogateType


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock provider instance."""
    provider = MagicMock(spec=BaseProvider)
    provider.answer.return_value = "original answer"
    return provider


@pytest.fixture
def explainer(mock_provider: MagicMock) -> LLMExplainer:
    """Initialize an LLM explainer with a mocked provider for fast testing."""
    with (
        patch(
            "xwhy.explainers.llm.ProviderResolver.resolve",
            return_value=mock_provider,
        ),
        patch("xwhy.explainers.llm.EmbeddingFactory"),
        patch("xwhy.explainers.llm.TextPerturbation"),
    ):
        return LLMExplainer(provider="openai", use_best_surrogate=True)


# ==========================================
# Initialization & Config Tests (__init__)
# ==========================================


@patch("xwhy.explainers.llm.EmbeddingType.from_str")
def test_init_raises_value_error_for_non_text_embedding(
    mock_from_str: MagicMock,
) -> None:
    """Ensure initialization fails early if the embedding is not for text."""
    mock_emb = MagicMock()
    mock_emb.is_text_embedding = False
    mock_emb.__str__.return_value = "invalid_embedding"  # type: ignore[attr-defined]
    mock_from_str.return_value = mock_emb

    with pytest.raises(ValueError, match="Invalid embedding type"):
        LLMExplainer(embedding_type="invalid")


@patch("xwhy.explainers.llm.EmbeddingFactory")
@patch("xwhy.explainers.llm.TextPerturbation")
def test_init_with_known_base_provider(
    mock_pert: MagicMock, mock_emb: MagicMock, mock_provider: MagicMock
) -> None:
    """Test init with a BaseProvider object that resolves to a valid ProviderType."""
    mock_provider.__class__.__name__ = "ValidProvider"
    with patch("xwhy.explainers.llm.ProviderType.from_str") as mock_from_str:
        mock_from_str.return_value = ProviderType.OPENAI
        explainer = LLMExplainer(provider=mock_provider)

        assert explainer.state.provider == mock_provider
        assert explainer.config.provider_type == ProviderType.OPENAI  # type: ignore[union-attr]
        mock_from_str.assert_called_with("valid")


@patch("xwhy.explainers.llm.EmbeddingFactory")
@patch("xwhy.explainers.llm.TextPerturbation")
def test_init_with_custom_base_provider(
    mock_pert: MagicMock, mock_emb: MagicMock, mock_provider: MagicMock
) -> None:
    """Test init with a custom BaseProvider triggering warning and fallback."""
    mock_provider.__class__.__name__ = "MyCustomProvider"

    with patch("xwhy.explainers.llm.logger.warning") as mock_warn:
        explainer = LLMExplainer(provider=mock_provider)

        assert explainer.state.provider == mock_provider
        assert explainer.config.provider_type == ProviderType.OPENAI  # type: ignore[union-attr]
        mock_warn.assert_called_once()
        assert "mapped to default config type" in mock_warn.call_args[0][0]


@patch("xwhy.explainers.llm.ProviderResolver.resolve")
@patch("xwhy.explainers.llm.EmbeddingFactory")
@patch("xwhy.explainers.llm.TextPerturbation")
def test_init_with_provider_enum(
    mock_pert: MagicMock, mock_emb: MagicMock, mock_resolve: MagicMock
) -> None:
    """Test init using a direct ProviderType Enum."""
    explainer = LLMExplainer(provider=ProviderType.OPENAI)
    assert explainer.config.provider_type == ProviderType.OPENAI  # type: ignore[union-attr]
    mock_resolve.assert_called_once()


@patch("xwhy.explainers.llm.ProviderResolver.resolve")
@patch("xwhy.explainers.llm.EmbeddingFactory")
@patch("xwhy.explainers.llm.TextPerturbation")
def test_init_with_provider_none(
    mock_pert: MagicMock, mock_emb: MagicMock, mock_resolve: MagicMock
) -> None:
    """Test init when provider is None (should fallback to OPENAI)."""
    explainer = LLMExplainer(provider=None)
    assert explainer.config.provider_type == ProviderType.OPENAI  # type: ignore[union-attr]
    mock_resolve.assert_called_once()


@patch("xwhy.explainers.llm.ProviderResolver.resolve")
@patch("xwhy.explainers.llm.EmbeddingFactory")
@patch("xwhy.explainers.llm.TextPerturbation")
def test_init_with_explicit_config(
    mock_pert: MagicMock, mock_emb: MagicMock, mock_resolve: MagicMock
) -> None:
    """Test init when explicitly passing an LLMConfig."""
    config = LLMConfig(provider_type=ProviderType.OPENAI)
    explainer = LLMExplainer(config=config)
    assert explainer.config == config
    mock_resolve.assert_called_once()


# ==========================================
# Run & Pipeline Execution Tests
# ==========================================


def test_run_raises_type_error_for_non_string_instance(explainer: LLMExplainer) -> None:
    """Test that run method raises TypeError when instance is not a string."""
    invalid_inputs = [123, ["prompt"], None, {"text": "hello"}]
    for invalid_input in invalid_inputs:
        with pytest.raises(
            TypeError, match=re.escape("LLMExplainer requires a string instance.")
        ):
            explainer.run(invalid_input)


def test_run_calls_explain_for_string_instance(explainer: LLMExplainer) -> None:
    """Test that run method delegates to explain correctly with valid string."""
    mock_result = MagicMock(spec=TextXWhyResult)
    with patch.object(explainer, "explain", return_value=mock_result) as mock_explain:
        instance = "test prompt"
        result = explainer.run(instance, extra_param=1)
        mock_explain.assert_called_once_with(instance, extra_param=1)
        assert result == mock_result


def test_explain_raises_type_error_for_non_string(explainer: LLMExplainer) -> None:
    """Test that explain raises TypeError for non-string inputs."""
    with pytest.raises(TypeError, match="requires the input prompt as a string"):
        explainer.explain(123)  # type: ignore[arg-type]


@pytest.mark.parametrize("missing_attr", ["provider", "embedding_model", "perturbator"])
def test_explain_raises_runtime_error_if_resources_missing(
    explainer: LLMExplainer, missing_attr: str
) -> None:
    """Test that missing any runtime resource raises a RuntimeError."""
    setattr(explainer.state, missing_attr, None)
    with pytest.raises(RuntimeError, match="runtime resources are not initialized"):
        explainer.explain("test prompt")


@patch("xwhy.explainers.llm.ProviderResolver.resolve")
@patch("xwhy.explainers.llm.TextPerturbation")
@patch("xwhy.explainers.llm.EmbeddingFactory")
@patch("xwhy.explainers.llm.WMDDistance")
@patch("xwhy.explainers.llm.DistanceNormalizer")
@patch("xwhy.explainers.llm.SurrogateTrainer")
@patch("xwhy.explainers.llm.SurrogateFactory")
@patch("xwhy.explainers.llm.RegressionMetrics")
def test_explain_success_best_surrogate(
    mock_metrics: MagicMock,
    mock_surrogate_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_normalizer: MagicMock,
    mock_wmd: MagicMock,
    mock_embedding_factory: MagicMock,
    mock_perturbation: MagicMock,
    mock_resolve: MagicMock,
    mock_provider: MagicMock,
) -> None:
    """Test the full explain pipeline when use_best_surrogate=True."""
    mock_resolve.return_value = mock_provider
    explainer = LLMExplainer(provider="openai", use_best_surrogate=True)

    mock_perturbation.return_value.generate.return_value = (
        ["res1"],
        [np.array([1, 0])],
    )
    mock_embedding_factory.create.return_value.load.return_value = MagicMock()
    mock_wmd.return_value.compute_batch.return_value = [("res1", 0.5)]
    mock_normalizer.min_max.return_value = [("val", 0.5)]

    mock_trainer.find_best.return_value = (SurrogateType.LIME, 0.9)
    mock_trainer.compute_weights.return_value = np.array([1.0])

    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.1])
    mock_surrogate.predict.return_value = np.array([0.5])
    mock_surrogate_factory.create.return_value = mock_surrogate

    result = explainer.explain("test prompt")

    assert isinstance(result, TextXWhyResult)
    assert mock_trainer.find_best.called
    assert "best_surrogate_method" in result.raw_data
    assert result.raw_data["best_surrogate_method"] == SurrogateType.LIME


@patch("xwhy.explainers.llm.ProviderResolver.resolve")
@patch("xwhy.explainers.llm.TextPerturbation")
@patch("xwhy.explainers.llm.EmbeddingFactory")
@patch("xwhy.explainers.llm.WMDDistance")
@patch("xwhy.explainers.llm.DistanceNormalizer")
@patch("xwhy.explainers.llm.SurrogateTrainer")
@patch("xwhy.explainers.llm.SurrogateFactory")
@patch("xwhy.explainers.llm.RegressionMetrics")
def test_explain_success_default_surrogate(
    mock_metrics: MagicMock,
    mock_surrogate_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_normalizer: MagicMock,
    mock_wmd: MagicMock,
    mock_embedding_factory: MagicMock,
    mock_perturbation: MagicMock,
    mock_resolve: MagicMock,
    mock_provider: MagicMock,
) -> None:
    """Test the full explain pipeline when using a default surrogate model."""
    mock_resolve.return_value = mock_provider
    explainer = LLMExplainer(
        provider="openai", use_best_surrogate=False, surrogate_type=SurrogateType.LIME
    )

    mock_perturbation.return_value.generate.return_value = (
        ["res1"],
        [np.array([1, 0])],
    )
    mock_embedding_factory.create.return_value.load.return_value = MagicMock()
    mock_wmd.return_value.compute_batch.return_value = [("res1", 0.5)]
    mock_normalizer.min_max.return_value = [("val", 0.5)]

    mock_trainer.compute_weights.return_value = np.array([1.0])

    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.1])
    mock_surrogate.predict.return_value = np.array([0.5])
    mock_surrogate_factory.create.return_value = mock_surrogate

    result = explainer.explain("test prompt")

    assert "surrogate_method" in result.raw_data
    assert result.raw_data["surrogate_method"] == SurrogateType.LIME
    assert not mock_trainer.find_best.called


@patch("xwhy.explainers.llm.TextXWhyResult.plot")
@patch("xwhy.explainers.llm.ProviderResolver.resolve")
@patch("xwhy.explainers.llm.TextPerturbation")
@patch("xwhy.explainers.llm.EmbeddingFactory")
@patch("xwhy.explainers.llm.WMDDistance")
@patch("xwhy.explainers.llm.DistanceNormalizer")
@patch("xwhy.explainers.llm.SurrogateTrainer")
@patch("xwhy.explainers.llm.SurrogateFactory")
@patch("xwhy.explainers.llm.RegressionMetrics")
def test_explain_fidelity_plot_flag(
    mock_metrics: MagicMock,
    mock_surrogate_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_normalizer: MagicMock,
    mock_wmd: MagicMock,
    mock_embedding_factory: MagicMock,
    mock_perturbation: MagicMock,
    mock_resolve: MagicMock,
    mock_plot: MagicMock,
    mock_provider: MagicMock,
) -> None:
    """Test that the fidelity_plot flag correctly triggers the plot method."""
    mock_resolve.return_value = mock_provider
    explainer = LLMExplainer(provider="openai", use_best_surrogate=False)

    mock_perturbation.return_value.generate.return_value = (
        ["res1"],
        [np.array([1, 0])],
    )
    mock_embedding_factory.create.return_value.load.return_value = MagicMock()
    mock_wmd.return_value.compute_batch.return_value = [("res1", 0.5)]
    mock_normalizer.min_max.return_value = [("val", 0.5)]

    mock_trainer.compute_weights.return_value = np.array([1.0])
    mock_surrogate_factory.create.return_value = MagicMock()

    # Test True
    explainer.explain("test prompt", fidelity_plot=True)
    mock_plot.assert_called_once_with(show=True)

    # Test False / Default
    mock_plot.reset_mock()
    explainer.explain("test prompt")
    mock_plot.assert_not_called()
