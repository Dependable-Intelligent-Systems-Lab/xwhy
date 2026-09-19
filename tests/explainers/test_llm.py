"""Tests for the LLM explainer module."""

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


def _wire_embedding_encode(mock_embedding_factory: MagicMock) -> MagicMock:
    """Attach a finite encode stub to the embedding factory mock.

    Args:
        mock_embedding_factory: Patched EmbeddingFactory class.

    Returns:
        MagicMock: The embedding model instance with encode configured.

    """
    embed_model = MagicMock()
    embed_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_embedding_factory.create.return_value.load.return_value = embed_model
    return embed_model


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


def test_init_raises_value_error_for_invalid_provider_string() -> None:
    """Ensure initialization fails if provider string is unknown."""
    provider = "unknown_provider_xyz"
    with pytest.raises(
        ValueError,
        match=f"'{provider}' is not a valid ProviderType. ",
    ):
        LLMExplainer(provider=provider)


@patch("xwhy.explainers.llm.ProviderResolver.resolve")
def test_initialize_raises_value_error_for_invalid_config_embedding(
    mock_resolve: MagicMock,
) -> None:
    """Ensure _initialize fails if an explicit config has non-text embedding."""
    mock_config = MagicMock()
    mock_config.provider_type = ProviderType.OPENAI
    mock_config.embedding_type.is_text_embedding = False
    mock_config.embedding_type.__str__.return_value = "fake_emb"

    with pytest.raises(ValueError, match="Invalid embedding type"):
        LLMExplainer(config=mock_config)


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
# Pipeline Execution Tests
# ==========================================


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
@patch("xwhy.explainers.llm.calculate_distance")
@patch("xwhy.explainers.llm.DistanceNormalizer")
@patch("xwhy.explainers.llm.SurrogateTrainer")
@patch("xwhy.explainers.llm.SurrogateFactory")
@patch("xwhy.explainers.llm.RegressionMetrics")
def test_explain_success_best_surrogate(
    mock_metrics: MagicMock,
    mock_surrogate_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_normalizer: MagicMock,
    mock_calc_dist: MagicMock,
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
    _wire_embedding_encode(mock_embedding_factory)
    mock_calc_dist.return_value = 0.5
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
@patch("xwhy.explainers.llm.calculate_distance")
@patch("xwhy.explainers.llm.DistanceNormalizer")
@patch("xwhy.explainers.llm.SurrogateTrainer")
@patch("xwhy.explainers.llm.SurrogateFactory")
@patch("xwhy.explainers.llm.RegressionMetrics")
def test_explain_success_default_surrogate(
    mock_metrics: MagicMock,
    mock_surrogate_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_normalizer: MagicMock,
    mock_calc_dist: MagicMock,
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
    _wire_embedding_encode(mock_embedding_factory)
    mock_calc_dist.return_value = 0.5
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
@patch("xwhy.explainers.llm.calculate_distance")
@patch("xwhy.explainers.llm.DistanceNormalizer")
@patch("xwhy.explainers.llm.SurrogateTrainer")
@patch("xwhy.explainers.llm.SurrogateFactory")
@patch("xwhy.explainers.llm.RegressionMetrics")
def test_explain_fidelity_plot_flag(
    mock_metrics: MagicMock,
    mock_surrogate_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_normalizer: MagicMock,
    mock_calc_dist: MagicMock,
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
    _wire_embedding_encode(mock_embedding_factory)
    mock_calc_dist.return_value = 0.5
    mock_normalizer.min_max.return_value = [("val", 0.5)]

    mock_trainer.compute_weights.return_value = np.array([1.0])
    mock_surrogate_factory.create.return_value = MagicMock()

    explainer.explain("test prompt", fidelity_plot=True)
    mock_plot.assert_called_once_with(show=True)

    mock_plot.reset_mock()
    explainer.explain("test prompt")
    mock_plot.assert_not_called()


@patch("xwhy.explainers.llm.ProviderResolver.resolve")
@patch("xwhy.explainers.llm.TextPerturbation")
@patch("xwhy.explainers.llm.EmbeddingFactory")
@patch("xwhy.explainers.llm.calculate_distance")
@patch("xwhy.explainers.llm.DistanceNormalizer")
@patch("xwhy.explainers.llm.SurrogateTrainer")
@patch("xwhy.explainers.llm.SurrogateFactory")
@patch("xwhy.explainers.llm.RegressionMetrics")
def test_llm_explain_filters_non_finite_distances(
    mock_metrics: MagicMock,
    mock_surrogate_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_normalizer: MagicMock,
    mock_calc_dist: MagicMock,
    mock_embedding_factory: MagicMock,
    mock_perturbation: MagicMock,
    mock_resolve: MagicMock,
    mock_provider: MagicMock,
) -> None:
    """Filter non-finite distances and train only on valid rows."""
    mock_resolve.return_value = mock_provider
    explainer = LLMExplainer(provider="openai", use_best_surrogate=False)

    mock_perturbation.return_value.generate.return_value = (
        ["res1", "res2", "res3"],
        [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])],
    )
    _wire_embedding_encode(mock_embedding_factory)

    mock_calc_dist.side_effect = [0.5, np.inf, 1.5]
    mock_normalizer.min_max.return_value = [
        ("val", 0.5),
        ("val", 1.0),
    ]

    mock_trainer.compute_weights.return_value = np.array([1.0, 1.0])
    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.1, 0.2])
    mock_surrogate.predict.return_value = np.array([0.5, 0.6])
    mock_surrogate_factory.create.return_value = mock_surrogate

    result = explainer.explain("test prompt")

    distances_used = result.raw_data["text_distances"]
    assert len(distances_used) == 2
    assert distances_used[0][1] == pytest.approx(0.5)
    assert distances_used[1][1] == pytest.approx(1.5)
    assert isinstance(result, TextXWhyResult)


@patch("xwhy.explainers.llm.ProviderResolver.resolve")
@patch("xwhy.explainers.llm.TextPerturbation")
@patch("xwhy.explainers.llm.EmbeddingFactory")
@patch("xwhy.explainers.llm.calculate_distance")
@patch("xwhy.explainers.llm.DistanceNormalizer")
@patch("xwhy.explainers.llm.SurrogateTrainer")
@patch("xwhy.explainers.llm.SurrogateFactory")
@patch("xwhy.explainers.llm.RegressionMetrics")
def test_llm_explain_raises_when_all_distances_non_finite(
    mock_metrics: MagicMock,
    mock_surrogate_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_normalizer: MagicMock,
    mock_calc_dist: MagicMock,
    mock_embedding_factory: MagicMock,
    mock_perturbation: MagicMock,
    mock_resolve: MagicMock,
    mock_provider: MagicMock,
) -> None:
    """Raise ValueError when every distance is non-finite."""
    mock_resolve.return_value = mock_provider
    explainer = LLMExplainer(provider="openai", use_best_surrogate=False)

    mock_perturbation.return_value.generate.return_value = (
        ["res1", "res2"],
        [np.array([1, 0]), np.array([0, 1])],
    )
    _wire_embedding_encode(mock_embedding_factory)

    mock_calc_dist.side_effect = [np.inf, np.nan]

    with pytest.raises(
        ValueError,
        match="All perturbations failed \\(0 valid distances\\)",
    ):
        explainer.explain("test prompt")


@patch("xwhy.explainers.llm.ProviderResolver.resolve")
@patch("xwhy.explainers.llm.TextPerturbation")
@patch("xwhy.explainers.llm.EmbeddingFactory")
@patch("xwhy.explainers.llm.calculate_distance")
@patch("xwhy.explainers.llm.DistanceNormalizer")
@patch("xwhy.explainers.llm.SurrogateTrainer")
@patch("xwhy.explainers.llm.SurrogateFactory")
@patch("xwhy.explainers.llm.RegressionMetrics")
@patch("xwhy.explainers.llm.logger")
def test_llm_explain_warns_on_low_valid_ratio(
    mock_logger: MagicMock,
    mock_metrics: MagicMock,
    mock_surrogate_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_normalizer: MagicMock,
    mock_calc_dist: MagicMock,
    mock_embedding_factory: MagicMock,
    mock_perturbation: MagicMock,
    mock_resolve: MagicMock,
    mock_provider: MagicMock,
) -> None:
    """Log a warning when valid ratio is below ``min_valid_ratio``."""
    mock_resolve.return_value = mock_provider
    explainer = LLMExplainer(
        provider="openai",
        use_best_surrogate=False,
        min_valid_ratio=0.5,
    )

    mock_perturbation.return_value.generate.return_value = (
        ["res1", "res2", "res3"],
        [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])],
    )
    _wire_embedding_encode(mock_embedding_factory)

    mock_calc_dist.side_effect = [0.5, np.inf, np.nan]
    mock_normalizer.min_max.return_value = [("val", 0.5)]

    mock_trainer.compute_weights.return_value = np.array([1.0])
    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.1, 0.2])
    mock_surrogate.predict.return_value = np.array([0.5])
    mock_surrogate_factory.create.return_value = mock_surrogate

    result = explainer.explain("test prompt")
    assert isinstance(result, TextXWhyResult)
    assert len(result.raw_data["text_distances"]) == 1

    warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
    assert any("Low valid perturbation ratio" in c for c in warning_calls)


@patch("xwhy.explainers.llm.ProviderResolver.resolve")
@patch("xwhy.explainers.llm.TextPerturbation")
@patch("xwhy.explainers.llm.EmbeddingFactory")
@patch("xwhy.explainers.llm.calculate_distance")
@patch("xwhy.explainers.llm.DistanceNormalizer")
@patch("xwhy.explainers.llm.SurrogateTrainer")
@patch("xwhy.explainers.llm.SurrogateFactory")
@patch("xwhy.explainers.llm.RegressionMetrics")
def test_llm_explain_includes_p_values(
    mock_metrics: MagicMock,
    mock_surrogate_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_normalizer: MagicMock,
    mock_calc_dist: MagicMock,
    mock_embedding_factory: MagicMock,
    mock_perturbation: MagicMock,
    mock_resolve: MagicMock,
    mock_provider: MagicMock,
) -> None:
    """Store filtered p-values when return_p_value is enabled."""
    mock_resolve.return_value = mock_provider
    explainer = LLMExplainer(
        provider="openai",
        use_best_surrogate=False,
        return_p_value=True,
        n_bootstrap=10,
    )

    mock_perturbation.return_value.generate.return_value = (
        ["res1", "res2"],
        [np.array([1, 0]), np.array([0, 1])],
    )
    _wire_embedding_encode(mock_embedding_factory)

    mock_calc_dist.return_value = (0.03, 0.5)
    mock_normalizer.min_max.return_value = [("val", 0.5), ("val", 0.6)]
    mock_trainer.compute_weights.return_value = np.array([1.0, 1.0])
    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.1, 0.2])
    mock_surrogate.predict.return_value = np.array([0.5, 0.6])
    mock_surrogate_factory.create.return_value = mock_surrogate

    result = explainer.explain("test prompt")

    assert "p_values" in result.raw_data
    assert list(result.raw_data["p_values"]) == pytest.approx([0.03, 0.03])


@patch("xwhy.explainers.llm.ProviderResolver.resolve")
@patch("xwhy.explainers.llm.TextPerturbation")
@patch("xwhy.explainers.llm.EmbeddingFactory")
@patch("xwhy.explainers.llm.calculate_distance")
@patch("xwhy.explainers.llm.DistanceNormalizer")
@patch("xwhy.explainers.llm.SurrogateTrainer")
@patch("xwhy.explainers.llm.SurrogateFactory")
@patch("xwhy.explainers.llm.RegressionMetrics")
def test_llm_explain_empty_embedding_fallback(
    mock_metrics: MagicMock,
    mock_surrogate_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_normalizer: MagicMock,
    mock_calc_dist: MagicMock,
    mock_embedding_factory: MagicMock,
    mock_perturbation: MagicMock,
    mock_resolve: MagicMock,
    mock_provider: MagicMock,
) -> None:
    """Use distance 1.0 when embeddings are empty arrays."""
    mock_resolve.return_value = mock_provider
    explainer = LLMExplainer(
        provider="openai", distance_type="wasserstein", use_best_surrogate=False
    )

    mock_perturbation.return_value.generate.return_value = (
        ["res1", "res2"],
        [np.array([1, 0]), np.array([0, 1])],
    )
    # Override the already-bound runtime embedding model
    embed_model = MagicMock()
    embed_model.encode.return_value = np.array([])
    explainer.state.embedding_model = embed_model

    mock_normalizer.min_max.return_value = [("val", 0.5), ("val", 0.6)]
    mock_trainer.compute_weights.return_value = np.array([1.0, 1.0])
    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.1, 0.2])
    mock_surrogate.predict.return_value = np.array([0.5, 0.6])
    mock_surrogate_factory.create.return_value = mock_surrogate

    result = explainer.explain("test prompt")

    distances_used = result.raw_data["text_distances"]
    assert len(distances_used) == 2
    assert distances_used[0][1] == pytest.approx(1.0)
    assert distances_used[1][1] == pytest.approx(1.0)
    mock_calc_dist.assert_not_called()


@patch("xwhy.explainers.llm.ProviderResolver.resolve")
@patch("xwhy.explainers.llm.TextPerturbation")
@patch("xwhy.explainers.llm.EmbeddingFactory")
@patch("xwhy.explainers.llm.calculate_distance")
@patch("xwhy.explainers.llm.DistanceNormalizer")
@patch("xwhy.explainers.llm.SurrogateTrainer")
@patch("xwhy.explainers.llm.SurrogateFactory")
@patch("xwhy.explainers.llm.RegressionMetrics")
def test_llm_explain_empty_embedding_with_p_values(
    mock_metrics: MagicMock,
    mock_surrogate_factory: MagicMock,
    mock_trainer: MagicMock,
    mock_normalizer: MagicMock,
    mock_calc_dist: MagicMock,
    mock_embedding_factory: MagicMock,
    mock_perturbation: MagicMock,
    mock_resolve: MagicMock,
    mock_provider: MagicMock,
) -> None:
    """Append NaN p-values when embeddings are empty and return_p_value."""
    mock_resolve.return_value = mock_provider
    explainer = LLMExplainer(
        provider="openai",
        distance_type="wasserstein",
        use_best_surrogate=False,
        return_p_value=True,
    )

    mock_perturbation.return_value.generate.return_value = (
        ["res1"],
        [np.array([1, 0])],
    )
    # Override the already-bound runtime embedding model
    embed_model = MagicMock()
    embed_model.encode.return_value = np.array([])
    explainer.state.embedding_model = embed_model

    mock_normalizer.min_max.return_value = [("val", 0.5)]
    mock_trainer.compute_weights.return_value = np.array([1.0])
    mock_surrogate = MagicMock()
    mock_surrogate.coefficients.return_value = np.array([0.1])
    mock_surrogate.predict.return_value = np.array([0.5])
    mock_surrogate_factory.create.return_value = mock_surrogate

    result = explainer.explain("test prompt")

    assert "p_values" in result.raw_data
    assert np.isnan(result.raw_data["p_values"][0])
