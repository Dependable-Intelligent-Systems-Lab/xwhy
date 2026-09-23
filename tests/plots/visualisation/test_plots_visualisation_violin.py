"""Tests for violin and layered-violin summary plots."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

matplotlib.use("Agg")

from xwhy.plots.visualisation.base import DimensionError
from xwhy.plots.visualisation.violin import (
    _shap_violin,
    _trim_crange,
    _violin_orientation_kwargs,
    violin,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rng(seed: int = 0) -> np.random.Generator:
    """Return a deterministic RNG."""
    return np.random.default_rng(seed)


def _shap_matrix(
    n_samples: int = 40,
    n_features: int = 4,
    seed: int = 0,
) -> NDArray[np.floating[Any]]:
    """Return a (n_samples, n_features) SHAP matrix with mixed signs."""
    rng = _rng(seed)
    return rng.normal(size=(n_samples, n_features)).astype(float)


def _feature_matrix(
    n_samples: int = 40,
    n_features: int = 4,
    seed: int = 1,
) -> NDArray[np.floating[Any]]:
    """Return a (n_samples, n_features) feature-value matrix."""
    rng = _rng(seed)
    return rng.uniform(-2.0, 2.0, size=(n_samples, n_features)).astype(float)


def _close_all() -> None:
    """Close every open matplotlib figure."""
    plt.close("all")


# ---------------------------------------------------------------------------
# _trim_crange
# ---------------------------------------------------------------------------


def test_trim_crange_normal() -> None:
    """Typical spread uses 5th/95th percentiles."""
    values = np.linspace(0.0, 100.0, 200)
    nan_mask = np.zeros(len(values), dtype=bool)
    vmin, vmax, cvals = _trim_crange(values, nan_mask)
    assert vmin < vmax
    assert cvals.shape == (len(values),)


def test_trim_crange_collapse_to_1_99() -> None:
    """When 5/95 collapse, fall back to 1/99 percentiles."""
    values = np.array([1.0] * 50 + [2.0])
    nan_mask = np.zeros(len(values), dtype=bool)
    vmin, vmax, cvals = _trim_crange(values, nan_mask)
    assert vmin <= vmax
    assert len(cvals) == int((~nan_mask).sum())


def test_trim_crange_collapse_to_min_max() -> None:
    """When all percentiles collapse, use raw min/max."""
    values = np.full(30, 3.14)
    nan_mask = np.zeros(len(values), dtype=bool)
    vmin, vmax, cvals = _trim_crange(values, nan_mask)
    assert vmin == vmax == 3.14
    assert len(cvals) == 30


def test_trim_crange_vmin_gt_vmax_guard() -> None:
    """Force vmin > vmax path then clamp to vmax."""
    # All-equal triggers collapse; the guard ``if vmin > vmax`` is defensive.
    values = np.array([5.0, 5.0, 5.0])
    nan_mask = np.zeros(3, dtype=bool)
    vmin, vmax, _cvals = _trim_crange(values, nan_mask)
    assert vmin <= vmax


def test_trim_crange_with_nans() -> None:
    """NaN entries are excluded from cvals via nan_mask."""
    values = np.array([1.0, np.nan, 3.0, 5.0, np.nan, 7.0])
    nan_mask = np.isnan(values)
    vmin, vmax, cvals = _trim_crange(values, nan_mask)
    assert len(cvals) == int((~nan_mask).sum())
    assert vmin <= vmax


def test_trim_crange_clips_outliers() -> None:
    """Values outside [vmin, vmax] are clipped in cvals."""
    values = np.concatenate([np.linspace(0, 10, 100), [1000.0, -1000.0]])
    nan_mask = np.zeros(len(values), dtype=bool)
    vmin, vmax, cvals = _trim_crange(values, nan_mask)
    assert float(cvals.max()) <= vmax + 1e-9
    assert float(cvals.min()) >= vmin - 1e-9


# ---------------------------------------------------------------------------
# _shap_violin — error paths
# ---------------------------------------------------------------------------


def test_shap_violin_rejects_list() -> None:
    """Multi-output list explanations raise TypeError."""
    with pytest.raises(TypeError, match="multi-output"):
        _shap_violin(shap_values=[np.zeros((10, 3)), np.zeros((10, 3))])


def test_shap_violin_rejects_bad_plot_type() -> None:
    """Unknown plot_type raises ValueError."""
    with pytest.raises(ValueError, match="plot_type"):
        _shap_violin(shap_values=_shap_matrix(), plot_type="bar")


def test_shap_violin_rejects_1d_vector() -> None:
    """1-D shap_values raise AssertionError."""
    with pytest.raises(AssertionError, match="matrix"):
        _shap_violin(shap_values=np.array([0.1, 0.2, 0.3]))


def test_shap_violin_plot_type_none_defaults() -> None:
    """plot_type=None falls back to violin."""
    try:
        _shap_violin(
            shap_values=_shap_matrix(20, 3),
            plot_type=None,  # type: ignore[arg-type]
        )
    finally:
        _close_all()


def test_shap_violin_dimension_error_extra_column() -> None:
    """shap_values has one more column than features → DimensionError."""
    sv = _shap_matrix(20, 5)
    feat = _feature_matrix(20, 4)
    with pytest.raises(DimensionError, match="shape"):
        _shap_violin(shap_values=sv, features=feat)


def test_shap_violin_dimension_error_mismatch() -> None:
    """Unrelated shape mismatch raises DimensionError."""
    sv = _shap_matrix(20, 4)
    feat = _feature_matrix(20, 2)
    with pytest.raises(DimensionError, match="shape"):
        _shap_violin(shap_values=sv, features=feat)


# ---------------------------------------------------------------------------
# _shap_violin — feature input variants
# ---------------------------------------------------------------------------


def test_shap_violin_features_dataframe() -> None:
    """DataFrame features supply column names when feature_names is None."""
    sv = _shap_matrix(30, 3)
    feat = pd.DataFrame(sv * 0.5, columns=["a", "b", "c"])
    try:
        _shap_violin(shap_values=sv, features=feat, max_display=3)
    finally:
        _close_all()


def test_shap_violin_features_dataframe_with_names() -> None:
    """DataFrame features with explicit feature_names keeps given names."""
    sv = _shap_matrix(30, 3)
    feat = pd.DataFrame(sv * 0.5, columns=["x", "y", "z"])
    try:
        _shap_violin(
            shap_values=sv,
            features=feat,
            feature_names=["x", "y", "z"],
            max_display=3,
        )
    finally:
        _close_all()


def test_shap_violin_features_list_as_names() -> None:
    """A list for features is treated as feature_names shorthand."""
    sv = _shap_matrix(25, 3)
    try:
        _shap_violin(
            shap_values=sv,
            features=["f0", "f1", "f2"],
            max_display=3,
        )
    finally:
        _close_all()


def test_shap_violin_features_list_with_names() -> None:
    """List features with explicit feature_names still clears features."""
    sv = _shap_matrix(25, 3)
    try:
        _shap_violin(
            shap_values=sv,
            features=["a", "b", "c"],
            feature_names=["a", "b", "c"],
            max_display=3,
        )
    finally:
        _close_all()


def test_shap_violin_features_1d_as_names() -> None:
    """1-D features array becomes feature_names when names are absent."""
    sv = _shap_matrix(25, 3)
    names_arr = np.array(["n0", "n1", "n2"], dtype=object)
    try:
        _shap_violin(shap_values=sv, features=names_arr, max_display=3)
    finally:
        _close_all()


def test_shap_violin_default_feature_names() -> None:
    """Missing feature_names yields 'Feature i' labels."""
    sv = _shap_matrix(20, 3)
    try:
        _shap_violin(shap_values=sv, features=None, max_display=3)
    finally:
        _close_all()


# ---------------------------------------------------------------------------
# _shap_violin — Explanation-like object
# ---------------------------------------------------------------------------


def test_shap_violin_explanation_object() -> None:
    """Objects whose type name ends with Explanation'> are unpacked."""

    class FakeExplanation:
        """Minimal stand-in whose type string matches SHAP Explanation."""

        def __init__(self) -> None:
            self.values = _shap_matrix(30, 3)
            self.data = _feature_matrix(30, 3)
            self.feature_names = ["e0", "e1", "e2"]

    # Force the type-string check used in the source.
    fake = FakeExplanation()
    original_type = type(fake)

    class Explanation(original_type):  # type: ignore[valid-type, misc]
        pass

    exp = Explanation()
    exp.values = fake.values
    exp.data = fake.data
    exp.feature_names = fake.feature_names
    assert str(type(exp)).endswith("Explanation'>")

    try:
        _shap_violin(shap_values=exp, max_display=3)
    finally:
        _close_all()


def test_shap_violin_explanation_with_external_features() -> None:
    """Explanation supplies values; caller still passes features."""

    class Explanation:
        def __init__(self) -> None:
            self.values = _shap_matrix(20, 3)
            self.data = None
            self.feature_names = None

    exp = Explanation()
    assert str(type(exp)).endswith("Explanation'>")
    feat = _feature_matrix(20, 3)
    try:
        _shap_violin(
            shap_values=exp,
            features=feat,
            feature_names=["a", "b", "c"],
            max_display=3,
        )
    finally:
        _close_all()


# ---------------------------------------------------------------------------
# _shap_violin — plot options
# ---------------------------------------------------------------------------


def test_shap_violin_title_warns() -> None:
    """Passing title emits DeprecationWarning."""
    sv = _shap_matrix(15, 3)
    try:
        with pytest.warns(DeprecationWarning, match="title"):
            _shap_violin(shap_values=sv, title="unused", max_display=3)
    finally:
        _close_all()


def test_shap_violin_use_log_scale() -> None:
    """use_log_scale sets a symmetric log x-axis."""
    sv = _shap_matrix(20, 3)
    try:
        _shap_violin(shap_values=sv, use_log_scale=True, max_display=3)
    finally:
        _close_all()


def test_shap_violin_max_display_none() -> None:
    """max_display=None defaults to 20."""
    sv = _shap_matrix(30, 5)
    try:
        _shap_violin(shap_values=sv, max_display=None)
    finally:
        _close_all()


def test_shap_violin_sort_false() -> None:
    """sort=False keeps original feature order (flipped)."""
    sv = _shap_matrix(20, 4)
    try:
        _shap_violin(shap_values=sv, sort=False, max_display=4)
    finally:
        _close_all()


def test_shap_violin_plot_size_tuple() -> None:
    """Tuple plot_size sets figure size explicitly."""
    sv = _shap_matrix(20, 3)
    try:
        _shap_violin(shap_values=sv, plot_size=(6.0, 4.0), max_display=3)
    finally:
        _close_all()


def test_shap_violin_plot_size_float() -> None:
    """Float plot_size scales row height."""
    sv = _shap_matrix(20, 3)
    try:
        _shap_violin(shap_values=sv, plot_size=0.5, max_display=3)
    finally:
        _close_all()


def test_shap_violin_plot_size_none() -> None:
    """plot_size=None leaves figure size untouched."""
    sv = _shap_matrix(15, 3)
    try:
        _shap_violin(shap_values=sv, plot_size=None, max_display=3)
    finally:
        _close_all()


def test_shap_violin_custom_color() -> None:
    """Explicit color is used when features are absent."""
    sv = _shap_matrix(20, 3)
    try:
        _shap_violin(shap_values=sv, color="#FF0000", max_display=3)
    finally:
        _close_all()


def test_shap_violin_cmap_none_uses_red_blue() -> None:
    """cmap=None selects the module RED_BLUE default."""
    sv = _shap_matrix(20, 3)
    feat = _feature_matrix(20, 3)
    try:
        _shap_violin(shap_values=sv, features=feat, cmap=None, max_display=3)
    finally:
        _close_all()


def test_shap_violin_rng_none() -> None:
    """rng=None creates a default Generator internally."""
    sv = _shap_matrix(20, 3)
    try:
        _shap_violin(shap_values=sv, rng=None, max_display=3)
    finally:
        _close_all()


# ---------------------------------------------------------------------------
# _shap_violin — violin with features (coloured)
# ---------------------------------------------------------------------------


def test_shap_violin_with_features_coloured() -> None:
    """Violin + features draws coloured density fills and scatter."""
    sv = _shap_matrix(50, 4, seed=2)
    feat = _feature_matrix(50, 4, seed=3)
    try:
        _shap_violin(
            shap_values=sv,
            features=feat,
            feature_names=["w", "x", "y", "z"],
            max_display=4,
            color_bar=True,
            rng=_rng(5),
        )
    finally:
        _close_all()


def test_shap_violin_low_std_adds_noise() -> None:
    """Near-constant SHAP column triggers KDE noise injection."""
    n = 40
    sv = np.zeros((n, 2))
    sv[:, 0] = 0.001  # near-zero variance
    sv[:, 1] = _rng(7).normal(size=n)
    feat = _feature_matrix(n, 2, seed=8)
    try:
        _shap_violin(
            shap_values=sv,
            features=feat,
            max_display=2,
            rng=_rng(9),
        )
    finally:
        _close_all()


def test_shap_violin_features_with_nans() -> None:
    """NaN feature values are plotted in grey; non-NaN are coloured."""
    n, f = 40, 3
    sv = _shap_matrix(n, f, seed=10)
    feat = _feature_matrix(n, f, seed=11)
    feat[0, 0] = np.nan
    feat[5, 1] = np.nan
    try:
        _shap_violin(
            shap_values=sv,
            features=feat,
            max_display=3,
            rng=_rng(12),
        )
    finally:
        _close_all()


def test_shap_violin_constant_feature_column() -> None:
    """Constant feature values exercise vmax == vmin colour scaling."""
    n, f = 30, 2
    sv = _shap_matrix(n, f, seed=13)
    feat = np.ones((n, f))
    try:
        _shap_violin(
            shap_values=sv,
            features=feat,
            max_display=2,
            rng=_rng(14),
        )
    finally:
        _close_all()


def test_shap_violin_color_bar_false() -> None:
    """color_bar=False skips the colour-bar legend."""
    sv = _shap_matrix(25, 3)
    feat = _feature_matrix(25, 3)
    try:
        _shap_violin(
            shap_values=sv,
            features=feat,
            color_bar=False,
            max_display=3,
        )
    finally:
        _close_all()


# ---------------------------------------------------------------------------
# _shap_violin — plain violin (no features)
# ---------------------------------------------------------------------------


def test_shap_violin_no_features() -> None:
    """Without features, matplotlib violinplot is used."""
    sv = _shap_matrix(30, 4)
    try:
        _shap_violin(shap_values=sv, features=None, max_display=4)
    finally:
        _close_all()


# ---------------------------------------------------------------------------
# _shap_violin — layered_violin
# ---------------------------------------------------------------------------


def test_shap_violin_layered_continuous() -> None:
    """Layered violin with many unique values uses fixed bins."""
    n, f = 60, 3
    sv = _shap_matrix(n, f, seed=20)
    # Continuous features → more uniques than max bins.
    feat = _feature_matrix(n, f, seed=21)
    try:
        _shap_violin(
            shap_values=sv,
            features=feat,
            plot_type="layered_violin",
            layered_violin_max_num_bins=5,
            color="coolwarm",
            max_display=3,
            rng=_rng(22),
        )
    finally:
        _close_all()


def test_shap_violin_layered_discrete() -> None:
    """Layered violin with few unique values uses value-based bins."""
    n, f = 50, 2
    sv = _shap_matrix(n, f, seed=23)
    # Discrete features with ≤ max bins unique values.
    feat = np.column_stack(
        [
            np.array([0, 1, 2] * (n // 3 + 1))[:n],
            np.array([10, 20] * (n // 2 + 1))[:n],
        ]
    ).astype(float)
    try:
        _shap_violin(
            shap_values=sv,
            features=feat,
            plot_type="layered_violin",
            layered_violin_max_num_bins=10,
            color="coolwarm",
            max_display=2,
            rng=_rng(24),
        )
    finally:
        _close_all()


def test_shap_violin_layered_single_sample_bin() -> None:
    """A bin with only one sample emits a warning and is skipped."""
    n, f = 8, 1
    sv = _shap_matrix(n, f, seed=25)
    # Force discrete bins where at least one value appears once.
    feat = np.array([[0], [0], [0], [1], [2], [2], [2], [3]], dtype=float)
    try:
        with pytest.warns(UserWarning, match="Not enough data in bin"):
            _shap_violin(
                shap_values=sv,
                features=feat,
                plot_type="layered_violin",
                layered_violin_max_num_bins=10,
                color="coolwarm",
                max_display=1,
                rng=_rng(26),
            )
    finally:
        _close_all()


def test_shap_violin_layered_color_not_colormap() -> None:
    """Non-colormap color string skips colour-bar for layered_violin."""
    n, f = 40, 2
    sv = _shap_matrix(n, f, seed=27)
    feat = _feature_matrix(n, f, seed=28)
    try:
        _shap_violin(
            shap_values=sv,
            features=feat,
            plot_type="layered_violin",
            color="#1E88E5",  # not a colormap name
            color_bar=True,
            max_display=2,
            rng=_rng(29),
        )
    finally:
        _close_all()


def test_shap_violin_layered_default_color() -> None:
    """color=None with layered_violin defaults to coolwarm."""
    n, f = 40, 2
    sv = _shap_matrix(n, f, seed=30)
    feat = _feature_matrix(n, f, seed=31)
    try:
        _shap_violin(
            shap_values=sv,
            features=feat,
            plot_type="layered_violin",
            color=None,
            max_display=2,
            rng=_rng(32),
        )
    finally:
        _close_all()


def test_shap_violin_layered_single_bin_first() -> None:
    """Single-sample first bin (i==0) skips without copying previous."""
    # One feature, values that create a leading singleton bin.
    sv = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]], dtype=float)
    feat = np.array([[0], [1], [1], [1], [1]], dtype=float)
    try:
        with pytest.warns(UserWarning, match="Not enough data in bin"):
            _shap_violin(
                shap_values=sv,
                features=feat,
                plot_type="layered_violin",
                layered_violin_max_num_bins=10,
                color="coolwarm",
                max_display=1,
                rng=_rng(33),
            )
    finally:
        _close_all()


# ---------------------------------------------------------------------------
# violin (public API)
# ---------------------------------------------------------------------------


def test_violin_show_false_returns_figure() -> None:
    """show=False returns a matplotlib Figure when finish is mocked."""
    sv = _shap_matrix(20, 3)
    mock_finish = MagicMock(return_value=plt.gcf())
    with (
        patch(
            "xwhy.plots.visualisation.violin._as_explanation",
            return_value=sv,
        ),
        patch(
            "xwhy.plots.visualisation.violin._finish_matplotlib",
            mock_finish,
        ),
        patch(
            "xwhy.plots.visualisation.violin._check_backend",
            return_value=None,
        ),
    ):
        result = violin(sv, show=False, max_display=3)
    assert result is not None
    mock_finish.assert_called_once()
    _close_all()


def test_violin_show_true() -> None:
    """show=True returns None via _finish_matplotlib."""
    sv = _shap_matrix(20, 3)
    mock_finish = MagicMock(return_value=None)
    with (
        patch(
            "xwhy.plots.visualisation.violin._as_explanation",
            return_value=sv,
        ),
        patch(
            "xwhy.plots.visualisation.violin._finish_matplotlib",
            mock_finish,
        ),
        patch(
            "xwhy.plots.visualisation.violin._check_backend",
            return_value=None,
        ),
    ):
        result = violin(sv, show=True, max_display=3)
    assert result is None
    mock_finish.assert_called_once()
    _close_all()


def test_violin_with_title_and_figsize() -> None:
    """Title sets suptitle; figsize creates a sized subplot."""
    sv = _shap_matrix(20, 3)
    feat = _feature_matrix(20, 3)
    mock_finish = MagicMock(side_effect=lambda fig, **_kw: fig)
    with (
        patch(
            "xwhy.plots.visualisation.violin._as_explanation",
            return_value=sv,
        ),
        patch(
            "xwhy.plots.visualisation.violin._finish_matplotlib",
            mock_finish,
        ),
        patch(
            "xwhy.plots.visualisation.violin._check_backend",
            return_value=None,
        ),
    ):
        result = violin(
            sv,
            show=False,
            title="My Violin",
            figsize=(7.0, 5.0),
            max_display=3,
            features=feat,
            feature_names=["x", "y", "z"],
        )
    assert result is not None
    _close_all()


def test_violin_max_display_none() -> None:
    """max_display=None is forwarded as 10 to _shap_violin."""
    sv = _shap_matrix(20, 3)
    mock_finish = MagicMock(side_effect=lambda fig, **_kw: fig)
    with (
        patch(
            "xwhy.plots.visualisation.violin._as_explanation",
            return_value=sv,
        ),
        patch(
            "xwhy.plots.visualisation.violin._finish_matplotlib",
            mock_finish,
        ),
        patch(
            "xwhy.plots.visualisation.violin._check_backend",
            return_value=None,
        ),
    ):
        result = violin(sv, show=False, max_display=None)
    assert result is not None
    _close_all()


def test_violin_kwargs_passthrough() -> None:
    """Keyword options reach _shap_violin (plot_type, sort, …)."""
    sv = _shap_matrix(25, 3)
    feat = _feature_matrix(25, 3)
    mock_finish = MagicMock(side_effect=lambda fig, **_kw: fig)
    with (
        patch(
            "xwhy.plots.visualisation.violin._as_explanation",
            return_value=sv,
        ),
        patch(
            "xwhy.plots.visualisation.violin._finish_matplotlib",
            mock_finish,
        ),
        patch(
            "xwhy.plots.visualisation.violin._check_backend",
            return_value=None,
        ),
    ):
        result = violin(
            sv,
            show=False,
            max_display=3,
            features=feat,
            feature_names=["a", "b", "c"],
            plot_type="violin",
            color="#00AA00",
            sort=False,
            color_bar=False,
            plot_size=(5.0, 3.0),
            use_log_scale=False,
            alpha=0.8,
        )
    assert result is not None
    _close_all()


def test_violin_layered_via_kwargs() -> None:
    """plot_type=layered_violin via kwargs on the public API."""
    sv = _shap_matrix(40, 2)
    feat = _feature_matrix(40, 2)
    mock_finish = MagicMock(side_effect=lambda fig, **_kw: fig)
    with (
        patch(
            "xwhy.plots.visualisation.violin._as_explanation",
            return_value=sv,
        ),
        patch(
            "xwhy.plots.visualisation.violin._finish_matplotlib",
            mock_finish,
        ),
        patch(
            "xwhy.plots.visualisation.violin._check_backend",
            return_value=None,
        ),
    ):
        result = violin(
            sv,
            show=False,
            max_display=2,
            features=feat,
            plot_type="layered_violin",
            color="coolwarm",
            layered_violin_max_num_bins=8,
        )
    assert result is not None
    _close_all()


def test_violin_as_explanation_none() -> None:
    """When _as_explanation is None the raw ndarray is used as-is."""
    sv = _shap_matrix(15, 2)
    mock_finish = MagicMock(side_effect=lambda fig, **_kw: fig)
    with (
        patch(
            "xwhy.plots.visualisation.violin._as_explanation",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.violin._finish_matplotlib",
            mock_finish,
        ),
        patch(
            "xwhy.plots.visualisation.violin._check_backend",
            return_value=None,
        ),
    ):
        result = violin(sv, show=False, max_display=2)
    assert result is not None
    _close_all()


def test_violin_finish_matplotlib_none_show_false() -> None:
    """Without _finish_matplotlib, show=False returns the figure."""
    sv = _shap_matrix(15, 2)
    with (
        patch(
            "xwhy.plots.visualisation.violin._as_explanation",
            return_value=sv,
        ),
        patch(
            "xwhy.plots.visualisation.violin._finish_matplotlib",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.violin._check_backend",
            return_value=None,
        ),
    ):
        result = violin(sv, show=False, max_display=2)
    assert result is not None
    assert hasattr(result, "savefig")
    _close_all()


def test_violin_finish_matplotlib_none_show_true() -> None:
    """Without _finish_matplotlib, show=True calls plt.show and returns None."""
    sv = _shap_matrix(15, 2)
    with (
        patch(
            "xwhy.plots.visualisation.violin._as_explanation",
            return_value=sv,
        ),
        patch(
            "xwhy.plots.visualisation.violin._finish_matplotlib",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.violin._check_backend",
            return_value=None,
        ),
        patch("xwhy.plots.visualisation.violin.plt.show") as mock_show,
    ):
        result = violin(sv, show=True, max_display=2)
    assert result is None
    mock_show.assert_called_once()
    _close_all()


def test_violin_check_backend_none_valid() -> None:
    """_check_backend is None and backend is matplotlib → ok."""
    sv = _shap_matrix(15, 2)
    mock_finish = MagicMock(side_effect=lambda fig, **_kw: fig)
    with (
        patch(
            "xwhy.plots.visualisation.violin._as_explanation",
            return_value=sv,
        ),
        patch(
            "xwhy.plots.visualisation.violin._finish_matplotlib",
            mock_finish,
        ),
        patch(
            "xwhy.plots.visualisation.violin._check_backend",
            new=None,
        ),
    ):
        result = violin(sv, show=False, backend="matplotlib", max_display=2)
    assert result is not None
    _close_all()


def test_violin_check_backend_none_invalid() -> None:
    """_check_backend is None and backend is unknown → ValueError."""
    sv = _shap_matrix(10, 2)
    with (
        patch(
            "xwhy.plots.visualisation.violin._as_explanation",
            return_value=sv,
        ),
        patch(
            "xwhy.plots.visualisation.violin._check_backend",
            new=None,
        ),
        pytest.raises(ValueError, match="Unsupported backend"),
    ):
        violin(sv, show=False, backend="plotly")


def test_violin_check_backend_called() -> None:
    """When _check_backend is present it is invoked."""
    sv = _shap_matrix(15, 2)
    mock_check = MagicMock()
    mock_finish = MagicMock(side_effect=lambda fig, **_kw: fig)
    with (
        patch(
            "xwhy.plots.visualisation.violin._as_explanation",
            return_value=sv,
        ),
        patch(
            "xwhy.plots.visualisation.violin._finish_matplotlib",
            mock_finish,
        ),
        patch(
            "xwhy.plots.visualisation.violin._check_backend",
            mock_check,
        ),
    ):
        violin(sv, show=False, backend="matplotlib", max_display=2)
    mock_check.assert_called_once()
    _close_all()


# ---------------------------------------------------------------------------
# Extra branch coverage helpers
# ---------------------------------------------------------------------------


def test_trim_crange_vmin_gt_vmax() -> None:
    """Force the defensive vmin > vmax clamp via mocked percentiles."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    nan_mask = np.zeros(5, dtype=bool)

    def fake_nanpercentile(arr: NDArray[Any], q: float, **_kwargs: Any) -> float:  # noqa: ANN401
        # Return inverted range so vmin > vmax after the first pair.
        if q in (5, 1):
            return 10.0
        if q in (95, 99):
            return 0.0
        return float(np.nanpercentile(arr, q))

    with patch(
        "xwhy.plots.visualisation.violin.np.nanpercentile",
        side_effect=fake_nanpercentile,
    ):
        vmin, vmax, cvals = _trim_crange(values, nan_mask)
    assert vmin <= vmax
    assert len(cvals) == 5


def test_shap_violin_explicit_cmap() -> None:
    """Passing an explicit cmap skips the cmap is None assignment."""
    sv = _shap_matrix(25, 3)
    feat = _feature_matrix(25, 3)
    try:
        _shap_violin(
            shap_values=sv,
            features=feat,
            cmap="viridis",
            max_display=3,
            rng=_rng(40),
        )
    finally:
        _close_all()


def test_violin_orientation_kwargs_new_mpl() -> None:
    """Matplotlib >= 3.10 uses the orientation keyword."""

    class NewV:
        def __ge__(self, other: object) -> bool:
            return True

        def __lt__(self, other: object) -> bool:
            return False

    with patch(
        "xwhy.plots.visualisation.violin.version.parse",
        return_value=NewV(),
    ):
        result = _violin_orientation_kwargs()
    assert result == {"orientation": "horizontal"}


def test_violin_orientation_kwargs_old_mpl() -> None:
    """Matplotlib < 3.10 uses the legacy vert flag."""

    class OldV:
        def __ge__(self, other: object) -> bool:
            return False

        def __lt__(self, other: object) -> bool:
            return True

    with patch(
        "xwhy.plots.visualisation.violin.version.parse",
        return_value=OldV(),
    ):
        result = _violin_orientation_kwargs()
    assert result == {"vert": False}


def test_shap_violin_layered_enters_branch() -> None:
    """Explicit layered_violin path with features and colormap color bar."""
    n, f = 50, 2
    sv = _shap_matrix(n, f, seed=50)
    feat = np.column_stack(
        [
            np.tile([0.0, 1.0, 2.0], n // 3 + 1)[:n],
            np.tile([5.0, 6.0], n // 2 + 1)[:n],
        ]
    )
    try:
        _shap_violin(
            shap_values=sv,
            features=feat,
            feature_names=["L0", "L1"],
            plot_type="layered_violin",
            layered_violin_max_num_bins=10,
            color="coolwarm",
            color_bar=True,
            max_display=2,
            rng=_rng(51),
        )
        assert len(plt.gca().collections) > 0 or len(plt.gca().patches) > 0
    finally:
        _close_all()


def test_shap_violin_layered_single_unique() -> None:
    """Layered violin with a constant feature (single bin, nbins == 1)."""
    n = 30
    sv = _shap_matrix(n, 1, seed=60)
    feat = np.zeros((n, 1))  # one unique value → single bin
    try:
        _shap_violin(
            shap_values=sv,
            features=feat,
            feature_names=["const"],
            plot_type="layered_violin",
            layered_violin_max_num_bins=10,
            color="coolwarm",
            color_bar=True,
            max_display=1,
            rng=_rng(61),
        )
    finally:
        _close_all()


def test_shap_violin_layered_continuous_many_uniques() -> None:
    """Layered violin else-branch: more uniques than max bins."""
    n, f = 80, 2
    rng = _rng(70)
    sv = rng.normal(size=(n, f)).astype(float)
    # All values unique → unique.shape[0] > layered_violin_max_num_bins
    feat = np.column_stack(
        [
            np.linspace(0.0, 1.0, n),
            np.linspace(2.0, 3.0, n),
        ]
    )
    try:
        _shap_violin(
            shap_values=sv,
            features=feat,
            feature_names=["c0", "c1"],
            plot_type="layered_violin",
            layered_violin_max_num_bins=5,
            color="coolwarm",
            color_bar=True,
            max_display=2,
            rng=_rng(71),
        )
        assert plt.gcf().axes  # plot was drawn
    finally:
        _close_all()


def test_shap_violin_layered_empty_feature_order() -> None:
    """Layered path with empty feature_order still reaches plt.xlim."""
    n, f = 20, 3
    sv = _shap_matrix(n, f, seed=72)
    feat = _feature_matrix(n, f, seed=73)
    try:
        _shap_violin(
            shap_values=sv,
            features=feat,
            plot_type="layered_violin",
            color="coolwarm",
            sort=False,
            max_display=0,  # → empty feature_order when sort=False
            rng=_rng(74),
        )
    finally:
        _close_all()


def test_shap_violin_layered_non_colormap_fill() -> None:
    """Layered fill uses raw color when name is not a colormap."""
    n, f = 40, 2
    sv = _shap_matrix(n, f, seed=75)
    feat = np.column_stack(
        [
            np.tile([0.0, 1.0], n // 2 + 1)[:n],
            np.tile([2.0, 3.0, 4.0], n // 3 + 1)[:n],
        ]
    )
    try:
        _shap_violin(
            shap_values=sv,
            features=feat,
            plot_type="layered_violin",
            color="#FF00AA",
            color_bar=True,  # skipped: color not in colormaps
            max_display=2,
            rng=_rng(76),
        )
    finally:
        _close_all()
