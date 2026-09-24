"""Test group_difference module."""

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

from xwhy.plots.visualisation.group_difference import (
    _shap_group_difference,
    group_difference,
)


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
    """Minimal explanation-like object for the public API."""

    values: NDArray[np.floating[Any]]
    feature_names: list[str] | None = None


def _make_shap(
    n_samples: int = 40,
    n_features: int = 5,
    seed: int = 0,
) -> NDArray[np.floating[Any]]:
    """Return a small deterministic attribution matrix."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_samples, n_features)).astype(float)


def _make_mask(n_samples: int = 40, seed: int = 1) -> NDArray[np.bool_]:
    """Return a balanced boolean group mask."""
    rng = np.random.default_rng(seed)
    return rng.random(n_samples) > 0.5


def _patch_as_explanation(exp: MockExplanation) -> Any:  # noqa: ANN401
    """Return a patch that makes ``_as_explanation`` return *exp*."""
    return patch(
        "xwhy.plots.visualisation.group_difference._as_explanation",
        return_value=exp,
    )


# ---------------------------------------------------------------------------
# _shap_group_difference
# ---------------------------------------------------------------------------


def test_shap_group_difference_basic() -> None:
    """2-D SHAP matrix with default feature names and sort."""
    shap = _make_shap()
    mask = _make_mask()
    plt.figure()
    _shap_group_difference(
        shap_values=shap,
        group_mask=mask,
        feature_names=None,
        show=False,
        rng=np.random.default_rng(0),
    )


def test_shap_group_difference_with_names() -> None:
    """Explicit feature names are used as y-tick labels."""
    shap = _make_shap(n_features=4)
    mask = _make_mask()
    names = ["a", "b", "c", "d"]
    plt.figure()
    _shap_group_difference(
        shap_values=shap,
        group_mask=mask,
        feature_names=names,
        show=False,
        rng=np.random.default_rng(0),
    )


def test_shap_group_difference_1d_vector() -> None:
    """1-D model-output vector is reshaped to a column matrix."""
    shap_1d = np.linspace(-1.0, 1.0, 30)
    mask = _make_mask(30)
    plt.figure()
    _shap_group_difference(
        shap_values=shap_1d,
        group_mask=mask,
        feature_names=None,
        show=False,
        rng=np.random.default_rng(0),
    )


def test_shap_group_difference_1d_with_names() -> None:
    """1-D vector with an explicit single feature name."""
    shap_1d = np.linspace(-0.5, 0.5, 20)
    mask = _make_mask(20)
    plt.figure()
    _shap_group_difference(
        shap_values=shap_1d,
        group_mask=mask,
        feature_names=["output"],
        show=False,
        rng=np.random.default_rng(0),
    )


def test_shap_group_difference_rng_none() -> None:
    """rng=None creates a default Generator."""
    shap = _make_shap(n_samples=20, n_features=3)
    mask = _make_mask(20)
    plt.figure()
    _shap_group_difference(
        shap_values=shap,
        group_mask=mask,
        feature_names=["x", "y", "z"],
        show=False,
        rng=None,
    )


def test_shap_group_difference_sort_false() -> None:
    """sort=False keeps natural feature order."""
    shap = _make_shap(n_features=4)
    mask = _make_mask()
    plt.figure()
    _shap_group_difference(
        shap_values=shap,
        group_mask=mask,
        feature_names=["a", "b", "c", "d"],
        sort=False,
        show=False,
        rng=np.random.default_rng(0),
    )


def test_shap_group_difference_max_display() -> None:
    """max_display truncates the number of feature rows."""
    shap = _make_shap(n_features=8)
    mask = _make_mask()
    plt.figure()
    _shap_group_difference(
        shap_values=shap,
        group_mask=mask,
        feature_names=[f"f{i}" for i in range(8)],
        max_display=3,
        show=False,
        rng=np.random.default_rng(0),
    )


def test_shap_group_difference_max_display_none() -> None:
    """max_display=None shows all features."""
    shap = _make_shap(n_features=3)
    mask = _make_mask()
    plt.figure()
    _shap_group_difference(
        shap_values=shap,
        group_mask=mask,
        feature_names=["a", "b", "c"],
        max_display=None,
        show=False,
        rng=np.random.default_rng(0),
    )


def test_shap_group_difference_with_ax() -> None:
    """Passing ax suppresses the internal show flag."""
    shap = _make_shap(n_features=3)
    mask = _make_mask()
    _fig, ax = plt.subplots(figsize=(6, 3))
    _shap_group_difference(
        shap_values=shap,
        group_mask=mask,
        feature_names=["a", "b", "c"],
        show=True,  # overridden to False because ax is given
        ax=ax,
        rng=np.random.default_rng(0),
    )


def test_shap_group_difference_custom_xlabel() -> None:
    """Custom xlabel is applied to the axes."""
    shap = _make_shap(n_features=3)
    mask = _make_mask()
    plt.figure()
    _shap_group_difference(
        shap_values=shap,
        group_mask=mask,
        feature_names=["a", "b", "c"],
        xlabel="Custom difference",
        show=False,
        rng=np.random.default_rng(0),
    )


def test_shap_group_difference_default_xlabel() -> None:
    """xlabel=None uses the default label string."""
    shap = _make_shap(n_features=2)
    mask = _make_mask()
    plt.figure()
    _shap_group_difference(
        shap_values=shap,
        group_mask=mask,
        feature_names=["a", "b"],
        xlabel=None,
        show=False,
        rng=np.random.default_rng(0),
    )


def test_shap_group_difference_xlim() -> None:
    """Xmin and xmax are forwarded to ax.set_xlim."""
    shap = _make_shap(n_features=3)
    mask = _make_mask()
    plt.figure()
    _shap_group_difference(
        shap_values=shap,
        group_mask=mask,
        feature_names=["a", "b", "c"],
        xmin=-2.0,
        xmax=2.0,
        show=False,
        rng=np.random.default_rng(0),
    )


def test_shap_group_difference_show_true() -> None:
    """show=True without ax calls plt.show()."""
    shap = _make_shap(n_samples=15, n_features=2)
    mask = _make_mask(15)
    with patch("matplotlib.pyplot.show") as mock_show:
        _shap_group_difference(
            shap_values=shap,
            group_mask=mask,
            feature_names=["a", "b"],
            show=True,
            ax=None,
            rng=np.random.default_rng(0),
        )
        mock_show.assert_called_once()


# ---------------------------------------------------------------------------
# group_difference (public entry-point)
# ---------------------------------------------------------------------------


def test_group_difference_with_finish() -> None:
    """When _finish_matplotlib is available, delegate to it."""
    shap = _make_shap()
    mask = _make_mask()
    exp = MockExplanation(values=shap, feature_names=["a", "b", "c", "d", "e"])
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.group_difference._finish_matplotlib",
            return_value="finished",
        ),
    ):
        result = group_difference(
            exp,
            mask,
            show=True,
            title="Group Diff",
            figsize=(6, 4),
            max_display=5,
            seed=0,
        )
        assert result == "finished"  # type: ignore[comparison-overlap]


def test_group_difference_show_true_no_finish() -> None:
    """show=True without _finish_matplotlib calls plt.show()."""
    shap = _make_shap(n_features=3)
    mask = _make_mask()
    exp = MockExplanation(values=shap, feature_names=["a", "b", "c"])
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.group_difference._finish_matplotlib",
            new=None,
        ),
        patch("matplotlib.pyplot.show") as mock_show,
    ):
        result = group_difference(exp, mask, show=True, seed=0)
        assert result is None
        mock_show.assert_called_once()


def test_group_difference_show_false_no_finish() -> None:
    """show=False without _finish_matplotlib returns Figure."""
    shap = _make_shap(n_features=3)
    mask = _make_mask()
    exp = MockExplanation(values=shap, feature_names=["a", "b", "c"])
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.group_difference._finish_matplotlib",
            new=None,
        ),
    ):
        fig = group_difference(exp, mask, show=False, seed=0)
        assert fig is not None
        assert isinstance(fig, Figure)


def test_group_difference_no_figsize() -> None:
    """figsize=None leaves ax=None so the helper sizes the figure."""
    shap = _make_shap(n_features=3)
    mask = _make_mask()
    exp = MockExplanation(values=shap, feature_names=["a", "b", "c"])
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.group_difference._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = group_difference(exp, mask, show=False, figsize=None, seed=0)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_group_difference_no_title() -> None:
    """title=None skips suptitle."""
    shap = _make_shap(n_features=3)
    mask = _make_mask()
    exp = MockExplanation(values=shap, feature_names=["a", "b", "c"])
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.group_difference._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = group_difference(exp, mask, show=False, title=None, seed=0)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_group_difference_kwargs_forwarded() -> None:
    """xlabel, xmin, xmax, sort kwargs reach _shap_group_difference."""
    shap = _make_shap(n_features=4)
    mask = _make_mask()
    exp = MockExplanation(values=shap, feature_names=["a", "b", "c", "d"])
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.group_difference._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = group_difference(
            exp,
            mask,
            show=False,
            figsize=(6, 3),
            seed=0,
            xlabel="Delta",
            xmin=-1.0,
            xmax=1.0,
            sort=False,
        )
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_group_difference_check_backend_none_valid() -> None:
    """_check_backend is None; valid backend is accepted."""
    shap = _make_shap(n_features=2)
    mask = _make_mask()
    exp = MockExplanation(values=shap, feature_names=["a", "b"])
    with (
        patch(
            "xwhy.plots.visualisation.group_difference._check_backend",
            new=None,
        ),
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.group_difference._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = group_difference(exp, mask, backend="matplotlib", show=False, seed=0)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_group_difference_check_backend_none_invalid() -> None:
    """_check_backend is None; invalid backend raises ValueError."""
    shap = _make_shap(n_features=2)
    mask = _make_mask()
    exp = MockExplanation(values=shap, feature_names=["a", "b"])
    with (
        patch(
            "xwhy.plots.visualisation.group_difference._check_backend",
            new=None,
        ),
        pytest.raises(ValueError, match="Unsupported backend"),
    ):
        group_difference(exp, mask, backend="plotly", show=False)


def test_group_difference_check_backend_called() -> None:
    """When _check_backend is present it is invoked."""
    shap = _make_shap(n_features=2)
    mask = _make_mask()
    exp = MockExplanation(values=shap, feature_names=["a", "b"])
    mock_check = MagicMock()
    with (
        patch(
            "xwhy.plots.visualisation.group_difference._check_backend",
            mock_check,
        ),
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.group_difference._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = group_difference(exp, mask, show=False, seed=0)
        assert result == "ok"  # type: ignore[comparison-overlap]
        mock_check.assert_called_once()


def test_group_difference_as_explanation_none() -> None:
    """_as_explanation is None; explanation is used as-is."""
    shap = _make_shap(n_features=2)
    mask = _make_mask()
    exp = MockExplanation(values=shap, feature_names=["a", "b"])
    with (
        patch(
            "xwhy.plots.visualisation.group_difference._as_explanation",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.group_difference._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = group_difference(exp, mask, show=False, seed=0)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_group_difference_as_explanation_called() -> None:
    """_as_explanation converts a raw object into an explanation."""
    raw = object()
    converted = MockExplanation(
        values=_make_shap(n_features=2),
        feature_names=["a", "b"],
    )
    mask = _make_mask()
    with (
        patch(
            "xwhy.plots.visualisation.group_difference._as_explanation",
            return_value=converted,
        ),
        patch(
            "xwhy.plots.visualisation.group_difference._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = group_difference(raw, mask, show=False, seed=0)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_group_difference_resolve_names_none() -> None:
    """_resolve_names is None; falls back to getattr feature_names."""
    shap = _make_shap(n_features=2)
    mask = _make_mask()
    exp = MockExplanation(values=shap, feature_names=["p", "q"])
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.group_difference._resolve_names",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.group_difference._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = group_difference(exp, mask, show=False, seed=0)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_group_difference_resolve_names_none_missing_attr() -> None:
    """_resolve_names is None and explanation has no feature_names."""

    class _Bare:
        values = _make_shap(n_features=2)

    bare = _Bare()
    mask = _make_mask()
    with (
        patch(
            "xwhy.plots.visualisation.group_difference._as_explanation",
            return_value=bare,
        ),
        patch(
            "xwhy.plots.visualisation.group_difference._resolve_names",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.group_difference._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = group_difference(bare, mask, show=False, seed=0)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_group_difference_resolve_names_called() -> None:
    """_resolve_names is not None; it supplies feature names."""
    shap = _make_shap(n_features=3)
    mask = _make_mask()
    exp = MockExplanation(values=shap, feature_names=None)
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.group_difference._resolve_names",
            return_value=["a", "b", "c"],
        ),
        patch(
            "xwhy.plots.visualisation.group_difference._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = group_difference(exp, mask, show=False, seed=0)
        assert result == "ok"  # type: ignore[comparison-overlap]
