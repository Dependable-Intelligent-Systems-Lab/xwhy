"""Tests for the LLM explainer module."""

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xwhy.core.result import TextXWhyResult
from xwhy.explainers.llm import LLMExplainer
from xwhy.surrogate.types import SurrogateType


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock provider instance."""
    provider = MagicMock()
    provider.answer.return_value = "original answer"
    return provider


@pytest.fixture
def explainer(mock_provider: MagicMock) -> LLMExplainer:
    """Initialize an LLM explainer with a mocked provider."""
    patcher = patch(
        "xwhy.explainers.llm.ProviderResolver.resolve",
        return_value=mock_provider,
    )
    with patcher:
        return LLMExplainer(provider="openai")


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
    explainer: LLMExplainer,
) -> None:
    """Test the full explain pipeline with use_best_surrogate=True."""
    # Setup mocks
    explainer.use_best_surrogate = True

    # Perturbation mock
    mock_perturbation.return_value.generate.return_value = (
        ["res1"],
        [np.array([1, 0])],
    )

    # Embedding mock
    mock_emb_model = MagicMock()
    mock_embedding_factory.create.return_value.load.return_value = mock_emb_model

    # WMD and Normalizer
    mock_wmd.return_value.compute_batch.return_value = np.array([0.5])
    mock_normalizer.min_max.return_value = [("val", 0.5)]

    # Trainer & Surrogate
    mock_trainer.find_best.return_value = (SurrogateType.LIME, 0.9)
    mock_trainer.compute_weights.return_value = np.array([1.0])

    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.1])
    mock_surrogate.predict.return_value = np.array([0.5])
    mock_surrogate_factory.create.return_value = mock_surrogate

    # Execute
    result = explainer.explain("test prompt")

    # Assertions
    assert isinstance(result, TextXWhyResult)
    assert mock_trainer.find_best.called
    assert "best_surrogate_method" in result.raw_data


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
    explainer: LLMExplainer,
) -> None:
    """Test the full explain pipeline with use_best_surrogate=False."""
    explainer.use_best_surrogate = False
    explainer.default_surrogate = SurrogateType.LIME

    # Setup basics
    mock_perturbation.return_value.generate.return_value = (
        ["res1"],
        [np.array([1, 0])],
    )

    mock_embedding_factory.create.return_value.load.return_value = MagicMock()
    mock_wmd.return_value.compute_batch.return_value = np.array([0.5])
    mock_normalizer.min_max.return_value = [("val", 0.5)]

    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.1])
    mock_surrogate_factory.create.return_value = mock_surrogate

    # Execute
    result = explainer.explain("test prompt")

    # Assertions
    assert "surrogate_method" in result.raw_data
    assert result.raw_data["surrogate_method"] == SurrogateType.LIME
    assert not mock_trainer.find_best.called  # Should not search for best


def test_run_invalid_input() -> None:
    """Ensure run raises TypeError for non-string inputs."""
    with patch("xwhy.explainers.llm.ProviderResolver.resolve"):
        explainer = LLMExplainer(provider="openai")
        with pytest.raises(TypeError, match="requires a string instance"):
            explainer.run(123)


def test_run_raises_type_error_for_non_string_instance(explainer: LLMExplainer) -> None:
    """Test that run method raises TypeError when instance is not a string."""
    invalid_inputs = [123, ["prompt"], None, {"text": "hello"}]

    for invalid_input in invalid_inputs:
        with pytest.raises(
            TypeError, match=re.escape("LLMExplainer requires a string instance.")
        ):
            explainer.run(invalid_input)


def test_run_calls_explain_for_string_instance(explainer: LLMExplainer) -> None:
    """Test that run method calls explain correctly with valid string."""
    mock_result = MagicMock(spec=TextXWhyResult)

    with patch.object(explainer, "explain", return_value=mock_result) as mock_explain:
        instance = "test prompt"
        result = explainer.run(instance, extra_param=1)

        mock_explain.assert_called_once_with(instance, extra_param=1)

        assert result == mock_result
