"""Test embedding module."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from numpy.typing import NDArray

from xwhy.plots.visualisation.embedding import _shap_embedding, embedding


@pytest.fixture(autouse=True)
def _close_plots() -> Any:  # noqa: ANN401
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class MockExplanation:
    """Minimal explanation-like object for the public embedding API."""

    values: NDArray[np.floating[Any]]
    feature_names: list[str] | None = None


def _make_shap(
    n_samples: int = 20,
    n_features: int = 4,
    seed: int = 0,
) -> NDArray[np.floating[Any]]:
    """Return a small deterministic attribution matrix."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_samples, n_features)).astype(float)


# ---------------------------------------------------------------------------
# _shap_embedding
# ---------------------------------------------------------------------------


def test_shap_embedding_default_feature_names() -> None:
    """feature_names=None generates Feature-i labels."""
    shap = _make_shap()
    with patch(
        "xwhy.plots.visualisation.embedding.convert_name",
        return_value=0,
    ):
        plt.figure()
        _shap_embedding(ind=0, shap_values=shap, feature_names=None, show=False)


def test_shap_embedding_sum_colouring() -> None:
    """Ind resolving to ``sum()`` colours by row-sum of SHAP values."""
    shap = _make_shap()
    with patch(
        "xwhy.plots.visualisation.embedding.convert_name",
        return_value="sum()",
    ):
        plt.figure()
        _shap_embedding(
            ind="sum()",
            shap_values=shap,
            feature_names=["a", "b", "c", "d"],
            show=False,
        )


def test_shap_embedding_feature_index_colouring() -> None:
    """Ind resolving to a column index colours by that feature's SHAP."""
    shap = _make_shap()
    with patch(
        "xwhy.plots.visualisation.embedding.convert_name",
        return_value=2,
    ):
        plt.figure()
        _shap_embedding(
            ind=2,
            shap_values=shap,
            feature_names=["a", "b", "c", "d"],
            show=False,
        )


def test_shap_embedding_pca_method() -> None:
    """method='pca' fits a 2-component PCA embedding."""
    shap = _make_shap()
    with patch(
        "xwhy.plots.visualisation.embedding.convert_name",
        return_value=0,
    ):
        plt.figure()
        _shap_embedding(
            ind=0,
            shap_values=shap,
            feature_names=["a", "b", "c", "d"],
            method="pca",
            show=False,
        )


def test_shap_embedding_precomputed_method() -> None:
    """Precomputed (n_samples, 2) array is used as the embedding."""
    shap = _make_shap(n_samples=15)
    precomputed = np.column_stack([np.linspace(-1, 1, 15), np.linspace(0, 1, 15)])
    with patch(
        "xwhy.plots.visualisation.embedding.convert_name",
        return_value=1,
    ):
        plt.figure()
        _shap_embedding(
            ind=1,
            shap_values=shap,
            feature_names=["a", "b", "c", "d"],
            method=precomputed,
            show=False,
        )


def test_shap_embedding_unsupported_method_string() -> None:
    """Unsupported string method raises ValueError."""
    shap = _make_shap()
    with (
        patch(
            "xwhy.plots.visualisation.embedding.convert_name",
            return_value=0,
        ),
        pytest.raises(ValueError, match="Unsupported embedding method"),
    ):
        _shap_embedding(
            ind=0,
            shap_values=shap,
            feature_names=["a", "b", "c", "d"],
            method="tsne",
            show=False,
        )


def test_shap_embedding_unsupported_method_bad_shape() -> None:
    """Array method with shape[1] != 2 raises ValueError."""
    shap = _make_shap(n_samples=10)
    bad = np.zeros((10, 3))
    with (
        patch(
            "xwhy.plots.visualisation.embedding.convert_name",
            return_value=0,
        ),
        pytest.raises(ValueError, match="Unsupported embedding method"),
    ):
        _shap_embedding(
            ind=0,
            shap_values=shap,
            feature_names=["a", "b", "c", "d"],
            method=bad,
            show=False,
        )


def test_shap_embedding_alpha() -> None:
    """Custom alpha is forwarded to scatter."""
    shap = _make_shap()
    with patch(
        "xwhy.plots.visualisation.embedding.convert_name",
        return_value=0,
    ):
        plt.figure()
        _shap_embedding(
            ind=0,
            shap_values=shap,
            feature_names=["a", "b", "c", "d"],
            alpha=0.4,
            show=False,
        )


# ---------------------------------------------------------------------------
# embedding (public entry-point)
# ---------------------------------------------------------------------------


def _patch_as_explanation(exp: MockExplanation) -> Any:  # noqa: ANN401
    """Return a patch that makes ``_as_explanation`` return *exp*."""
    return patch(
        "xwhy.plots.visualisation.embedding._as_explanation",
        return_value=exp,
    )


def test_embedding_with_finish_matplotlib() -> None:
    """When _finish_matplotlib is available, delegate to it.

    Also covers figsize is not None and title is set.
    """
    exp = MockExplanation(values=_make_shap(), feature_names=["a", "b", "c", "d"])
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.embedding.convert_name",
            return_value=0,
        ),
        patch(
            "xwhy.plots.visualisation.embedding._finish_matplotlib",
            return_value="finished",
        ),
    ):
        result = embedding(
            ind=0,
            explanation=exp,
            show=True,
            title="Embedding",
            figsize=(6, 4),
        )
        assert result == "finished"  # type: ignore[comparison-overlap]


def test_embedding_show_true_no_finish() -> None:
    """show=True without _finish_matplotlib calls plt.show() and returns None."""
    exp = MockExplanation(values=_make_shap(), feature_names=["a", "b", "c", "d"])
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.embedding.convert_name",
            return_value=0,
        ),
        patch(
            "xwhy.plots.visualisation.embedding._finish_matplotlib",
            new=None,
        ),
        patch("matplotlib.pyplot.show") as mock_show,
    ):
        result = embedding(ind=0, explanation=exp, show=True)
        assert result is None
        mock_show.assert_called_once()


def test_embedding_show_false_no_finish() -> None:
    """show=False without _finish_matplotlib returns the Figure."""
    exp = MockExplanation(values=_make_shap(), feature_names=["a", "b", "c", "d"])
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.embedding.convert_name",
            return_value=0,
        ),
        patch(
            "xwhy.plots.visualisation.embedding._finish_matplotlib",
            new=None,
        ),
    ):
        fig = embedding(ind=0, explanation=exp, show=False)
        assert fig is not None
        assert isinstance(fig, Figure)


def test_embedding_no_figsize() -> None:
    """figsize=None skips plt.subplots(figsize=...)."""
    exp = MockExplanation(values=_make_shap(), feature_names=["a", "b", "c", "d"])
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.embedding.convert_name",
            return_value=0,
        ),
        patch(
            "xwhy.plots.visualisation.embedding._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = embedding(ind=0, explanation=exp, show=False, figsize=None)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_embedding_no_title() -> None:
    """title=None skips fig.suptitle."""
    exp = MockExplanation(values=_make_shap(), feature_names=["a", "b", "c", "d"])
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.embedding.convert_name",
            return_value=0,
        ),
        patch(
            "xwhy.plots.visualisation.embedding._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = embedding(ind=0, explanation=exp, show=False, title=None)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_embedding_sum_via_public_api() -> None:
    """Public API with ind='sum()' and method/alpha kwargs."""
    exp = MockExplanation(values=_make_shap(), feature_names=["w", "x", "y", "z"])
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.embedding.convert_name",
            return_value="sum()",
        ),
        patch(
            "xwhy.plots.visualisation.embedding._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = embedding(
            ind="sum()",
            explanation=exp,
            show=False,
            method="pca",
            alpha=0.8,
        )
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_embedding_precomputed_via_kwargs() -> None:
    """Precomputed embedding passed through kwargs['method']."""
    shap = _make_shap(n_samples=12)
    exp = MockExplanation(values=shap, feature_names=["a", "b", "c", "d"])
    precomputed = np.random.default_rng(1).normal(size=(12, 2))
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.embedding.convert_name",
            return_value=0,
        ),
        patch(
            "xwhy.plots.visualisation.embedding._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = embedding(
            ind=0,
            explanation=exp,
            show=False,
            method=precomputed,
        )
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_embedding_check_backend_none_valid() -> None:
    """_check_backend is None; valid backend is accepted."""
    exp = MockExplanation(values=_make_shap(), feature_names=["a", "b", "c", "d"])
    with (
        patch(
            "xwhy.plots.visualisation.embedding._check_backend",
            new=None,
        ),
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.embedding.convert_name",
            return_value=0,
        ),
        patch(
            "xwhy.plots.visualisation.embedding._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = embedding(
            ind=0,
            explanation=exp,
            backend="matplotlib",
            show=False,
        )
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_embedding_check_backend_none_invalid() -> None:
    """_check_backend is None; invalid backend raises ValueError."""
    exp = MockExplanation(values=_make_shap())
    with (
        patch(
            "xwhy.plots.visualisation.embedding._check_backend",
            new=None,
        ),
        pytest.raises(ValueError, match="Unsupported backend"),
    ):
        embedding(
            ind=0,
            explanation=exp,
            backend="plotly",
            show=False,
        )


def test_embedding_check_backend_called() -> None:
    """When _check_backend is present it is invoked with allowed set."""
    exp = MockExplanation(values=_make_shap(), feature_names=["a", "b", "c", "d"])
    mock_check = MagicMock()
    with (
        patch(
            "xwhy.plots.visualisation.embedding._check_backend",
            mock_check,
        ),
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.embedding.convert_name",
            return_value=0,
        ),
        patch(
            "xwhy.plots.visualisation.embedding._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = embedding(ind=0, explanation=exp, show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]
        mock_check.assert_called_once()
        assert mock_check.call_args[0][0] == "matplotlib"
        assert "matplotlib" in mock_check.call_args[0][1]


def test_embedding_as_explanation_none() -> None:
    """_as_explanation is None; explanation is used as-is."""
    exp = MockExplanation(values=_make_shap(), feature_names=["a", "b", "c", "d"])
    with (
        patch(
            "xwhy.plots.visualisation.embedding._as_explanation",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.embedding.convert_name",
            return_value=0,
        ),
        patch(
            "xwhy.plots.visualisation.embedding._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = embedding(ind=0, explanation=exp, show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_embedding_as_explanation_called() -> None:
    """_as_explanation is not None; it converts the explanation."""
    raw = object()
    converted = MockExplanation(
        values=_make_shap(),
        feature_names=["a", "b", "c", "d"],
    )
    with (
        patch(
            "xwhy.plots.visualisation.embedding._as_explanation",
            return_value=converted,
        ),
        patch(
            "xwhy.plots.visualisation.embedding.convert_name",
            return_value=0,
        ),
        patch(
            "xwhy.plots.visualisation.embedding._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = embedding(ind=0, explanation=raw, show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_embedding_resolve_names_none() -> None:
    """_resolve_names is None; falls back to getattr feature_names (line 126)."""
    exp = MockExplanation(values=_make_shap(), feature_names=["p", "q", "r", "s"])
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.embedding._resolve_names",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.embedding.convert_name",
            return_value=0,
        ),
        patch(
            "xwhy.plots.visualisation.embedding._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = embedding(ind=0, explanation=exp, show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_embedding_resolve_names_none_missing_attr() -> None:
    """_resolve_names is None and explanation has no feature_names."""

    class _Bare:
        values = _make_shap()

    bare = _Bare()
    with (
        patch(
            "xwhy.plots.visualisation.embedding._as_explanation",
            return_value=bare,
        ),
        patch(
            "xwhy.plots.visualisation.embedding._resolve_names",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.embedding.convert_name",
            return_value=0,
        ),
        patch(
            "xwhy.plots.visualisation.embedding._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = embedding(ind=0, explanation=bare, show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_embedding_resolve_names_called() -> None:
    """_resolve_names is not None; it supplies feature names."""
    exp = MockExplanation(values=_make_shap(), feature_names=None)
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.embedding._resolve_names",
            return_value=["a", "b", "c", "d"],
        ),
        patch(
            "xwhy.plots.visualisation.embedding.convert_name",
            return_value=0,
        ),
        patch(
            "xwhy.plots.visualisation.embedding._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = embedding(ind=0, explanation=exp, show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]
