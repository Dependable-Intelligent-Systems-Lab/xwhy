"""Test heatmap module."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from numpy.typing import NDArray

from xwhy.plots.visualisation.heatmap import (
    _collapse_excess_features,
    _order_instances,
    _resolve_instance_order,
    _shap_heatmap,
    heatmap,
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
    """Minimal multi-row explanation for the heatmap API."""

    values: NDArray[np.floating[Any]]
    feature_names: list[str] | None = None


def _make_values(
    n_instances: int = 20,
    n_features: int = 6,
    seed: int = 0,
) -> NDArray[np.floating[Any]]:
    """Return a small deterministic attribution matrix."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_instances, n_features)).astype(float)


def _patch_as_explanation(exp: MockExplanation) -> Any:  # noqa: ANN401
    """Return a patch that makes ``_as_explanation`` return *exp*."""
    return patch(
        "xwhy.plots.visualisation.heatmap._as_explanation",
        return_value=exp,
    )


# ---------------------------------------------------------------------------
# _order_instances
# ---------------------------------------------------------------------------


def test_order_instances_output() -> None:
    """strategy='output' sorts by row-sum of attributions."""
    values = _make_values(10, 4)
    order = _order_instances(values, "output")
    assert order.shape == (10,)
    sums = values.sum(axis=1)
    np.testing.assert_array_equal(order, np.argsort(sums))


def test_order_instances_hclust_small() -> None:
    """strategy='hclust' with 2 < n <= 2000 uses linkage."""
    values = _make_values(15, 4)
    order = _order_instances(values, "hclust")
    assert order.shape == (15,)
    assert set(order.tolist()) == set(range(15))


def test_order_instances_hclust_too_few() -> None:
    """Hclust with n_instances <= 2 falls back to output ordering."""
    values = _make_values(2, 3)
    order = _order_instances(values, "hclust")
    np.testing.assert_array_equal(order, np.argsort(values.sum(axis=1)))


def test_order_instances_hclust_too_many() -> None:
    """Hclust with n_instances > 2000 falls back to output ordering."""
    values = np.zeros((2001, 2))
    order = _order_instances(values, "hclust")
    np.testing.assert_array_equal(order, np.argsort(values.sum(axis=1)))


def test_order_instances_hclust_linkage_fails() -> None:
    """Linkage failure falls back to output ordering."""
    values = _make_values(10, 3)
    with patch(
        "xwhy.plots.visualisation.heatmap.linkage",
        side_effect=ValueError("boom"),
    ):
        order = _order_instances(values, "hclust")
    np.testing.assert_array_equal(order, np.argsort(values.sum(axis=1)))


def test_order_instances_none() -> None:
    """strategy='none' (or any other) returns natural order."""
    values = _make_values(8, 3)
    order = _order_instances(values, "none")
    np.testing.assert_array_equal(order, np.arange(8))


# ---------------------------------------------------------------------------
# _resolve_instance_order
# ---------------------------------------------------------------------------


def test_resolve_instance_order_ndarray() -> None:
    """Ndarray instance_order is returned unchanged."""
    values = _make_values(5, 3)
    idx = np.array([2, 0, 1, 4, 3])
    result = _resolve_instance_order(values, idx)
    np.testing.assert_array_equal(result, idx)


def test_resolve_instance_order_hclust_in_range() -> None:
    """Hclust with mid-size matrix uses optimal leaf ordering."""
    values = _make_values(12, 4)
    order = _resolve_instance_order(values, "hclust")
    assert order.shape == (12,)
    assert set(order.tolist()) == set(range(12))


def test_resolve_instance_order_hclust_out_of_range() -> None:
    """Hclust with n <= 2 falls back to output sort."""
    values = _make_values(2, 3)
    order = _resolve_instance_order(values, "hclust")
    np.testing.assert_array_equal(order, np.argsort(values.sum(axis=1)))


def test_resolve_instance_order_hclust_fails() -> None:
    """pdist/hierarchy failure falls back to output sort."""
    values = _make_values(10, 3)
    with patch(
        "xwhy.plots.visualisation.heatmap.scipy.spatial.distance.pdist",
        side_effect=ValueError("fail"),
    ):
        order = _resolve_instance_order(values, "hclust")
    np.testing.assert_array_equal(order, np.argsort(values.sum(axis=1)))


def test_resolve_instance_order_string_output() -> None:
    """String 'output' delegates to _order_instances."""
    values = _make_values(8, 3)
    order = _resolve_instance_order(values, "output")
    np.testing.assert_array_equal(order, np.argsort(values.sum(axis=1)))


def test_resolve_instance_order_other() -> None:
    """Non-str, non-ndarray falls through to np.asarray."""
    values = _make_values(4, 2)
    order = _resolve_instance_order(values, [3, 1, 0, 2])  # type: ignore[arg-type]
    np.testing.assert_array_equal(order, np.array([3, 1, 0, 2]))


# ---------------------------------------------------------------------------
# _collapse_excess_features
# ---------------------------------------------------------------------------


def test_collapse_no_excess() -> None:
    """When n_features <= max_display, arrays are returned unchanged."""
    values = _make_values(5, 4)
    fvals = np.abs(values).mean(0)
    names = np.array(["a", "b", "c", "d"])
    v, fv, n = _collapse_excess_features(values, fvals, names, max_display=10)
    np.testing.assert_array_equal(v, values)
    np.testing.assert_array_equal(fv, fvals)
    assert n == ["a", "b", "c", "d"]


def test_collapse_excess() -> None:
    """Excess features are summed into a final 'Sum of N other' column."""
    values = _make_values(5, 8)
    fvals = np.abs(values).mean(0)
    names = np.array([f"f{i}" for i in range(8)])
    v, fv, n = _collapse_excess_features(values, fvals, names, max_display=4)
    assert v.shape == (5, 4)
    assert fv.shape == (4,)
    assert len(n) == 4
    assert "Sum of" in n[-1]
    # First 3 columns preserved
    np.testing.assert_allclose(v[:, :3], values[:, :3])
    np.testing.assert_allclose(v[:, -1], values[:, 3:].sum(1))


# ---------------------------------------------------------------------------
# _shap_heatmap
# ---------------------------------------------------------------------------


def test_shap_heatmap_basic() -> None:
    """Default heatmap with feature names and hclust order."""
    exp = MockExplanation(
        values=_make_values(15, 6),
        feature_names=[f"f{i}" for i in range(6)],
    )
    plt.figure()
    ax = _shap_heatmap(exp, instance_order="hclust", max_display=6, show=False)
    assert ax is not None


def test_shap_heatmap_no_feature_names() -> None:
    """Missing feature_names generates Feature-i labels."""
    exp = MockExplanation(values=_make_values(10, 4), feature_names=None)
    plt.figure()
    ax = _shap_heatmap(exp, instance_order="output", max_display=4, show=False)
    assert ax is not None


def test_shap_heatmap_cmap_none() -> None:
    """cmap=None falls back to RED_WHITE_BLUE."""
    exp = MockExplanation(
        values=_make_values(8, 4),
        feature_names=["a", "b", "c", "d"],
    )
    plt.figure()
    ax = _shap_heatmap(exp, cmap=None, max_display=4, show=False)
    assert ax is not None


def test_shap_heatmap_custom_cmap() -> None:
    """Explicit colormap is used."""
    exp = MockExplanation(
        values=_make_values(8, 3),
        feature_names=["a", "b", "c"],
    )
    cmap = LinearSegmentedColormap.from_list("rwb", ["#f00", "#fff", "#00f"])
    plt.figure()
    ax = _shap_heatmap(exp, cmap=cmap, max_display=3, show=False)
    assert ax is not None


def test_shap_heatmap_feature_values_provided() -> None:
    """Explicit feature_values skip the mean-|SHAP| default."""
    values = _make_values(10, 4)
    exp = MockExplanation(values=values, feature_names=["a", "b", "c", "d"])
    fvals = np.array([4.0, 3.0, 2.0, 1.0])
    plt.figure()
    ax = _shap_heatmap(
        exp,
        feature_values=fvals,
        max_display=4,
        show=False,
    )
    assert ax is not None


def test_shap_heatmap_feature_order_provided() -> None:
    """Explicit feature_order is respected."""
    exp = MockExplanation(
        values=_make_values(10, 4),
        feature_names=["a", "b", "c", "d"],
    )
    plt.figure()
    ax = _shap_heatmap(
        exp,
        feature_order=np.array([3, 1, 0, 2]),
        max_display=4,
        show=False,
    )
    assert ax is not None


def test_shap_heatmap_with_ax() -> None:
    """Drawing onto a supplied axes skips internal size adjustment."""
    exp = MockExplanation(
        values=_make_values(8, 3),
        feature_names=["a", "b", "c"],
    )
    _fig, ax = plt.subplots(figsize=(6, 4))
    result = _shap_heatmap(exp, ax=ax, max_display=3, show=False)
    assert result is ax


def test_shap_heatmap_ax_none() -> None:
    """ax=None uses gcf/gca and sets figure size."""
    exp = MockExplanation(
        values=_make_values(8, 3),
        feature_names=["a", "b", "c"],
    )
    plt.figure()
    ax = _shap_heatmap(exp, ax=None, plot_width=7.0, max_display=3, show=False)
    assert ax is not None


def test_shap_heatmap_collapse() -> None:
    """max_display smaller than n_features collapses excess columns."""
    exp = MockExplanation(
        values=_make_values(12, 10),
        feature_names=[f"f{i}" for i in range(10)],
    )
    plt.figure()
    ax = _shap_heatmap(exp, max_display=4, show=False)
    assert ax is not None


def test_shap_heatmap_zero_fx() -> None:
    """All-zero values: fx_max == 0 skips normalization."""
    exp = MockExplanation(
        values=np.zeros((5, 3)),
        feature_names=["a", "b", "c"],
    )
    plt.figure()
    ax = _shap_heatmap(exp, max_display=3, show=False)
    assert ax is not None


def test_shap_heatmap_zero_feature_values() -> None:
    """Zero feature_values: bar_widths become zeros."""
    exp = MockExplanation(
        values=_make_values(8, 3),
        feature_names=["a", "b", "c"],
    )
    plt.figure()
    ax = _shap_heatmap(
        exp,
        feature_values=np.zeros(3),
        max_display=3,
        show=False,
    )
    assert ax is not None


def test_shap_heatmap_instance_order_none() -> None:
    """instance_order='none' keeps natural instance order."""
    exp = MockExplanation(
        values=_make_values(8, 3),
        feature_names=["a", "b", "c"],
    )
    plt.figure()
    ax = _shap_heatmap(exp, instance_order="none", max_display=3, show=False)
    assert ax is not None


def test_shap_heatmap_instance_order_array() -> None:
    """Precomputed instance index array is accepted."""
    values = _make_values(6, 3)
    exp = MockExplanation(values=values, feature_names=["a", "b", "c"])
    order = np.array([5, 0, 1, 2, 3, 4])
    plt.figure()
    ax = _shap_heatmap(exp, instance_order=order, max_display=3, show=False)
    assert ax is not None


# ---------------------------------------------------------------------------
# heatmap (public entry-point)
# ---------------------------------------------------------------------------


def test_heatmap_with_finish() -> None:
    """When _finish_matplotlib is available, delegate to it."""
    exp = MockExplanation(
        values=_make_values(12, 5),
        feature_names=[f"f{i}" for i in range(5)],
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.heatmap._finish_matplotlib",
            return_value="finished",
        ),
    ):
        result = heatmap(
            exp,
            show=True,
            title="Heatmap",
            figsize=(8, 5),
            max_display=5,
        )
        assert result == "finished"  # type: ignore[comparison-overlap]


def test_heatmap_show_true_no_finish() -> None:
    """show=True without _finish_matplotlib calls plt.show()."""
    exp = MockExplanation(
        values=_make_values(8, 3),
        feature_names=["a", "b", "c"],
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.heatmap._finish_matplotlib",
            new=None,
        ),
        patch("matplotlib.pyplot.show") as mock_show,
    ):
        result = heatmap(exp, show=True, max_display=3)
        assert result is None
        mock_show.assert_called_once()


def test_heatmap_show_false_no_finish() -> None:
    """show=False without _finish_matplotlib returns Figure."""
    exp = MockExplanation(
        values=_make_values(8, 3),
        feature_names=["a", "b", "c"],
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.heatmap._finish_matplotlib",
            new=None,
        ),
    ):
        fig = heatmap(exp, show=False, max_display=3)
        assert fig is not None
        assert isinstance(fig, Figure)


def test_heatmap_no_figsize() -> None:
    """figsize=None creates a figure sized from max_display."""
    exp = MockExplanation(
        values=_make_values(8, 3),
        feature_names=["a", "b", "c"],
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.heatmap._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = heatmap(exp, show=False, figsize=None, max_display=3)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_heatmap_no_title() -> None:
    """title=None skips suptitle."""
    exp = MockExplanation(
        values=_make_values(8, 3),
        feature_names=["a", "b", "c"],
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.heatmap._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = heatmap(exp, show=False, title=None, max_display=3)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_heatmap_max_display_none() -> None:
    """max_display=None defaults to 10."""
    exp = MockExplanation(
        values=_make_values(8, 4),
        feature_names=["a", "b", "c", "d"],
    )
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.heatmap._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = heatmap(exp, show=False, max_display=None)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_heatmap_kwargs_forwarded() -> None:
    """feature_values, feature_order, cmap, plot_width reach _shap_heatmap."""
    values = _make_values(10, 4)
    exp = MockExplanation(values=values, feature_names=["a", "b", "c", "d"])
    cmap = LinearSegmentedColormap.from_list("rwb", ["#f00", "#fff", "#00f"])
    with (
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.heatmap._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = heatmap(
            exp,
            show=False,
            figsize=(7, 4),
            max_display=4,
            feature_values=np.array([1.0, 2.0, 3.0, 4.0]),
            feature_order=np.array([3, 2, 1, 0]),
            cmap=cmap,
            plot_width=7.0,
            instance_order="output",
        )
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_heatmap_check_backend_none_valid() -> None:
    """_check_backend is None; valid backend is accepted."""
    exp = MockExplanation(
        values=_make_values(6, 3),
        feature_names=["a", "b", "c"],
    )
    with (
        patch(
            "xwhy.plots.visualisation.heatmap._check_backend",
            new=None,
        ),
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.heatmap._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = heatmap(exp, backend="matplotlib", show=False, max_display=3)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_heatmap_check_backend_none_invalid() -> None:
    """_check_backend is None; invalid backend raises ValueError."""
    exp = MockExplanation(values=_make_values(4, 2), feature_names=["a", "b"])
    with (
        patch(
            "xwhy.plots.visualisation.heatmap._check_backend",
            new=None,
        ),
        pytest.raises(ValueError, match="Unsupported backend"),
    ):
        heatmap(exp, backend="plotly", show=False)


def test_heatmap_check_backend_called() -> None:
    """When _check_backend is present it is invoked."""
    exp = MockExplanation(
        values=_make_values(6, 3),
        feature_names=["a", "b", "c"],
    )
    mock_check = MagicMock()
    with (
        patch(
            "xwhy.plots.visualisation.heatmap._check_backend",
            mock_check,
        ),
        _patch_as_explanation(exp),
        patch(
            "xwhy.plots.visualisation.heatmap._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = heatmap(exp, show=False, max_display=3)
        assert result == "ok"  # type: ignore[comparison-overlap]
        mock_check.assert_called_once()


def test_heatmap_as_explanation_none() -> None:
    """_as_explanation is None; explanation is used as-is."""
    exp = MockExplanation(
        values=_make_values(6, 3),
        feature_names=["a", "b", "c"],
    )
    with (
        patch(
            "xwhy.plots.visualisation.heatmap._as_explanation",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.heatmap._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = heatmap(exp, show=False, max_display=3)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_heatmap_as_explanation_called() -> None:
    """_as_explanation converts a raw object into an explanation."""
    raw = object()
    converted = MockExplanation(
        values=_make_values(6, 3),
        feature_names=["a", "b", "c"],
    )
    with (
        patch(
            "xwhy.plots.visualisation.heatmap._as_explanation",
            return_value=converted,
        ),
        patch(
            "xwhy.plots.visualisation.heatmap._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = heatmap(raw, show=False, max_display=3)
        assert result == "ok"  # type: ignore[comparison-overlap]
