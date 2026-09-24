"""Test monitoring module."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from numpy.typing import NDArray

from xwhy.plots.visualisation.monitoring import (
    _feature_index,
    _shap_monitoring,
    monitoring,
    truncate_text,
)


@pytest.fixture(autouse=True)
def _close_plots() -> Any:  # noqa: ANN401
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_values(
    n_samples: int = 200,
    n_features: int = 4,
    seed: int = 0,
) -> NDArray[np.floating[Any]]:
    """Return a deterministic attribution matrix."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_samples, n_features)).astype(float)


def _make_features(
    n_samples: int = 200,
    n_features: int = 4,
    seed: int = 1,
) -> NDArray[np.floating[Any]]:
    """Return a deterministic feature matrix."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_samples, n_features)).astype(float)


def _make_explanation(
    n_samples: int = 200,
    n_features: int = 4,
    *,
    with_data: bool = True,
    seed: int = 0,
) -> SimpleNamespace:
    """Build a minimal explanation-like object."""
    values = _make_values(n_samples, n_features, seed=seed)
    data = _make_features(n_samples, n_features, seed=seed + 1) if with_data else None
    names = [f"f{i}" for i in range(n_features)]
    return SimpleNamespace(values=values, data=data, feature_names=names)


def _patch_as_explanation(exp: object) -> Any:  # noqa: ANN401
    """Return a patch that makes ``_as_explanation`` return *exp*."""
    return patch(
        "xwhy.plots.visualisation.monitoring._as_explanation",
        return_value=exp,
    )


# ---------------------------------------------------------------------------
# truncate_text
# ---------------------------------------------------------------------------


def test_truncate_text_short() -> None:
    """Short strings are returned unchanged."""
    assert truncate_text("hello", 30) == "hello"


def test_truncate_text_long() -> None:
    """Long strings are truncated with a middle ellipsis."""
    text = "abcdefghijklmnopqrstuvwxyz"
    result = truncate_text(text, 10)
    assert "..." in result
    assert len(result) <= 10 + 3  # ellipsis expands slightly by design


# ---------------------------------------------------------------------------
# _feature_index
# ---------------------------------------------------------------------------


def test_feature_index_none_with_global_importance() -> None:
    """ind=None uses _global_importance when available."""
    values = _make_values(20, 3)
    names = ["a", "b", "c"]
    with patch(
        "xwhy.plots.visualisation.monitoring._global_importance",
        return_value=np.array([0.1, 0.9, 0.2]),
    ):
        idx = _feature_index(None, names, values)
    assert idx == 1


def test_feature_index_none_without_global_importance() -> None:
    """ind=None falls back to mean-|SHAP| ranking."""
    values = np.array([[1.0, 0.0, 0.5], [1.0, 0.0, 0.5]])
    names = ["a", "b", "c"]
    with patch(
        "xwhy.plots.visualisation.monitoring._global_importance",
        new=None,
    ):
        idx = _feature_index(None, names, values)
    assert idx == 0  # highest mean |value|


def test_feature_index_by_name() -> None:
    """String feature name resolves to its column index."""
    values = _make_values(10, 3)
    names = ["alpha", "beta", "gamma"]
    assert _feature_index("beta", names, values) == 1


def test_feature_index_unknown_name() -> None:
    """Unknown feature name raises ValueError with a preview."""
    values = _make_values(10, 3)
    names = ["a", "b", "c"]
    with pytest.raises(ValueError, match="Unknown feature"):
        _feature_index("missing", names, values)


def test_feature_index_unknown_name_long_preview() -> None:
    """Preview truncates when there are more than 10 names."""
    values = _make_values(5, 12)
    names = [f"f{i}" for i in range(12)]
    with pytest.raises(ValueError, match=r"\.\.\."):
        _feature_index("nope", names, values)


def test_feature_index_by_int() -> None:
    """Integer index is validated and returned (with wrap for negatives)."""
    values = _make_values(10, 4)
    names = ["a", "b", "c", "d"]
    assert _feature_index(2, names, values) == 2
    assert _feature_index(-1, names, values) == 3


def test_feature_index_out_of_range() -> None:
    """Out-of-range integer index raises ValueError."""
    values = _make_values(10, 3)
    names = ["a", "b", "c"]
    with pytest.raises(ValueError, match="out of range"):
        _feature_index(10, names, values)
    with pytest.raises(ValueError, match="out of range"):
        _feature_index(-10, names, values)


# ---------------------------------------------------------------------------
# _shap_monitoring
# ---------------------------------------------------------------------------


def test_shap_monitoring_basic() -> None:
    """Basic monitoring scatter with default feature names."""
    shap = _make_values(200, 3)
    feats = _make_features(200, 3)
    _shap_monitoring(ind=0, shap_values=shap, features=feats, show=False)


def test_shap_monitoring_dataframe() -> None:
    """DataFrame features supply column names when feature_names is None."""
    shap = _make_values(120, 3)
    df = pd.DataFrame(
        _make_features(120, 3),
        columns=["x", "y", "z"],
    )
    _shap_monitoring(
        ind=1,
        shap_values=shap,
        features=df,
        feature_names=None,
        show=False,
    )


def test_shap_monitoring_dataframe_with_names() -> None:
    """DataFrame with explicit feature_names keeps the provided names."""
    shap = _make_values(120, 2)
    df = pd.DataFrame(_make_features(120, 2), columns=["a", "b"])
    _shap_monitoring(
        ind=0,
        shap_values=shap,
        features=df,
        feature_names=["custom_a", "custom_b"],
        show=False,
    )


def test_shap_monitoring_default_names() -> None:
    """feature_names=None on an array generates Feature-i labels."""
    shap = _make_values(100, 2)
    feats = _make_features(100, 2)
    _shap_monitoring(
        ind=0,
        shap_values=shap,
        features=feats,
        feature_names=None,
        show=False,
    )


def test_shap_monitoring_significant_pvalue() -> None:
    """A clear change-point draws the dashed axvline (min_pval threshold).

    Small within-group noise avoids scipy's catastrophic-cancellation
    RuntimeWarning from ``ttest_ind`` on identical samples.
    """
    n = 200
    rng = np.random.default_rng(0)
    shap = np.zeros((n, 2))
    shap[: n // 2, 0] = rng.normal(0.0, 0.1, n // 2)
    shap[n // 2 :, 0] = rng.normal(10.0, 0.1, n - n // 2)
    feats = rng.normal(size=(n, 2))
    _shap_monitoring(ind=0, shap_values=shap, features=feats, show=False)


def test_shap_monitoring_few_samples_no_pvals() -> None:
    """Fewer than 2*inc samples yields empty pvals (min_pval=1.0)."""
    shap = _make_values(30, 2)  # inc=50, so range is empty
    feats = _make_features(30, 2)
    _shap_monitoring(ind=0, shap_values=shap, features=feats, show=False)


def test_shap_monitoring_show_true() -> None:
    """show=True calls plt.show()."""
    shap = _make_values(100, 2)
    feats = _make_features(100, 2)
    with patch("matplotlib.pyplot.show") as mock_show:
        _shap_monitoring(ind=0, shap_values=shap, features=feats, show=True)
        mock_show.assert_called_once()


# ---------------------------------------------------------------------------
# monitoring (public)
# ---------------------------------------------------------------------------


def test_monitoring_with_finish() -> None:
    """When _finish_matplotlib is available, delegate to it."""
    exp = _make_explanation()
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.monitoring._finish_matplotlib",
            return_value="finished",
        ),
    ):
        result = monitoring(
            0,
            exp,
            show=True,
            title="Monitor",
            figsize=(10, 3),
        )
        assert result == "finished"  # type: ignore[comparison-overlap]


def test_monitoring_show_true_no_finish() -> None:
    """show=True without _finish_matplotlib calls plt.show()."""
    exp = _make_explanation(n_samples=100, n_features=3)
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.monitoring._finish_matplotlib",
            new=None,
        ),
        patch("matplotlib.pyplot.show") as mock_show,
    ):
        result = monitoring(0, exp, show=True)
        assert result is None
        mock_show.assert_called_once()


def test_monitoring_show_false_no_finish() -> None:
    """show=False without _finish_matplotlib returns Figure."""
    exp = _make_explanation(n_samples=100, n_features=3)
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.monitoring._finish_matplotlib",
            new=None,
        ),
    ):
        fig = monitoring(0, exp, show=False)
        assert fig is not None
        assert isinstance(fig, Figure)


def test_monitoring_by_name() -> None:
    """Feature name is resolved through _feature_index."""
    exp = _make_explanation(n_samples=100, n_features=3)
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.monitoring._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = monitoring("f1", exp, show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_monitoring_features_from_data() -> None:
    """features=None uses explanation.data when present."""
    exp = _make_explanation(with_data=True)
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.monitoring._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = monitoring(0, exp, features=None, show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_monitoring_features_zeros_fallback() -> None:
    """features=None and no data falls back to a zero matrix."""
    exp = _make_explanation(with_data=False)
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.monitoring._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = monitoring(0, exp, features=None, show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_monitoring_explicit_features() -> None:
    """Explicit features matrix is forwarded to _shap_monitoring."""
    exp = _make_explanation(n_samples=80, n_features=3)
    feats = _make_features(80, 3)
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.monitoring._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = monitoring(0, exp, features=feats, show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_monitoring_no_figsize_no_title() -> None:
    """figsize=None and title=None skip those branches."""
    exp = _make_explanation(n_samples=80, n_features=2)
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.monitoring._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = monitoring(0, exp, show=False, figsize=None, title=None)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_monitoring_check_backend_none_valid() -> None:
    """_check_backend is None; valid backend is accepted."""
    exp = _make_explanation(n_samples=80, n_features=2)
    with (
        patch(
            "xwhy.plots.visualisation.monitoring._check_backend",
            new=None,
        ),
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.monitoring._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = monitoring(0, exp, backend="matplotlib", show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_monitoring_check_backend_none_invalid() -> None:
    """_check_backend is None; invalid backend raises ValueError."""
    exp = _make_explanation(n_samples=40, n_features=2)
    with (
        patch(
            "xwhy.plots.visualisation.monitoring._check_backend",
            new=None,
        ),
        pytest.raises(ValueError, match="Unsupported backend"),
    ):
        monitoring(0, exp, backend="plotly", show=False)


def test_monitoring_check_backend_called() -> None:
    """When _check_backend is present it is invoked."""
    exp = _make_explanation(n_samples=80, n_features=2)
    mock_check = MagicMock()
    with (
        patch(
            "xwhy.plots.visualisation.monitoring._check_backend",
            mock_check,
        ),
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.monitoring._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = monitoring(0, exp, show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]
        mock_check.assert_called_once()


def test_monitoring_as_explanation_none() -> None:
    """_as_explanation is None; explanation is used as-is."""
    exp = _make_explanation(n_samples=80, n_features=2)
    with (
        patch(
            "xwhy.plots.visualisation.monitoring._as_explanation",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.monitoring._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = monitoring(0, exp, show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_monitoring_resolve_names_none() -> None:
    """_resolve_names is None; falls back to getattr feature_names."""
    exp = _make_explanation(n_samples=80, n_features=2)
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.monitoring._resolve_names",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.monitoring._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = monitoring(0, exp, show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_monitoring_resolve_names_none_missing_attr() -> None:
    """_resolve_names is None and explanation has no feature_names."""
    values = _make_values(80, 2)
    exp = SimpleNamespace(values=values, data=_make_features(80, 2))
    with (
        patch(
            "xwhy.plots.visualisation.monitoring._as_explanation",
            return_value=exp,
        ),
        patch(
            "xwhy.plots.visualisation.monitoring._resolve_names",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.monitoring._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = monitoring(0, exp, show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_monitoring_resolve_names_called() -> None:
    """_resolve_names is not None; it supplies feature names."""
    exp = _make_explanation(n_samples=80, n_features=2)
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.monitoring._resolve_names",
            return_value=["x", "y"],
        ),
        patch(
            "xwhy.plots.visualisation.monitoring._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = monitoring("x", exp, show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]
