"""Test decision module."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import Any
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import LinearSegmentedColormap
from numpy.typing import NDArray

from xwhy.plots.visualisation.decision import (
    DecisionPlotResult,
    IdentityLink,
    Link,
    LogitLink,
    _change_shap_base_value,
    _decision_plot_matplotlib,
    _shap_decision,
    convert_to_link,
    decision,
    hclust_ordering,
)


@pytest.fixture(autouse=True)
def _close_plots() -> Any:  # noqa: ANN401
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shap(
    n_obs: int = 2,
    n_feat: int = 4,
    seed: int = 0,
) -> NDArray[np.floating[Any]]:
    """Return a small deterministic SHAP matrix."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_obs, n_feat)).astype(float)


def _make_interaction(
    n_obs: int = 1,
    n_feat: int = 3,
    seed: int = 1,
) -> NDArray[np.floating[Any]]:
    """Return a small interaction cube (n_obs, n_feat, n_feat)."""
    rng = np.random.default_rng(seed)
    cube = rng.normal(size=(n_obs, n_feat, n_feat)).astype(float)
    # Symmetrise for realism
    for i in range(n_obs):
        cube[i] = (cube[i] + cube[i].T) / 2
    return cube


# ---------------------------------------------------------------------------
# Link classes
# ---------------------------------------------------------------------------


def test_link_str() -> None:
    """Base Link.__str__ returns the class name."""
    assert str(Link()) == "Link"


def test_identity_link_str() -> None:
    """IdentityLink string form is 'identity'."""
    assert str(IdentityLink()) == "identity"


def test_identity_link_f_and_finv() -> None:
    """IdentityLink is a no-op for scalar and array."""
    link = IdentityLink()
    assert link.f(0.5) == 0.5
    assert link.finv(0.5) == 0.5
    arr = np.array([0.1, 0.9])
    np.testing.assert_array_equal(link.f(arr), arr)
    np.testing.assert_array_equal(link.finv(arr), arr)


def test_logit_link_str() -> None:
    """LogitLink string form is 'logit'."""
    assert str(LogitLink()) == "logit"


def test_logit_link_f() -> None:
    """LogitLink.f maps probability to log-odds with clipping."""
    result = LogitLink.f(0.5)
    assert isinstance(result, float | np.floating)
    assert abs(float(result)) < 1e-10
    # Extreme values are clipped
    extreme = LogitLink.f(0.0)
    assert np.isfinite(extreme)


def test_logit_link_finv() -> None:
    """LogitLink.finv maps log-odds back to probability."""
    result = LogitLink.finv(0.0)
    assert abs(float(result) - 0.5) < 1e-10
    arr = np.array([0.0, 1.0])
    inv = LogitLink.finv(arr)
    assert inv.shape == arr.shape  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# convert_to_link
# ---------------------------------------------------------------------------


def test_convert_to_link_instance() -> None:
    """Passing an existing Link returns it unchanged."""
    link = IdentityLink()
    assert convert_to_link(link) is link


def test_convert_to_link_identity_str() -> None:
    """String 'identity' yields IdentityLink."""
    assert isinstance(convert_to_link("identity"), IdentityLink)


def test_convert_to_link_logit_str() -> None:
    """String 'logit' yields LogitLink."""
    assert isinstance(convert_to_link("logit"), LogitLink)


def test_convert_to_link_invalid() -> None:
    """Unrecognised value raises TypeError."""
    with pytest.raises(TypeError, match="subclass of Link"):
        convert_to_link("unknown")


# ---------------------------------------------------------------------------
# hclust_ordering
# ---------------------------------------------------------------------------


def test_hclust_ordering() -> None:
    """hclust_ordering returns a permutation of feature indices."""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [1.1, 2.1, 3.1]])
    order = hclust_ordering(x)
    assert order.shape == (3,)
    assert set(order.tolist()) == {0, 1, 2}


def test_hclust_ordering_anchor_first_unused() -> None:
    """anchor_first is accepted but unused (API compatibility)."""
    x = np.eye(4)
    order = hclust_ordering(x, anchor_first=True)
    assert order.shape == (4,)


# ---------------------------------------------------------------------------
# _change_shap_base_value
# ---------------------------------------------------------------------------


def test_change_shap_base_value_2d() -> None:
    """2-D SHAP values are shifted uniformly by the base delta."""
    shap = np.ones((2, 4))
    out = _change_shap_base_value(1.0, 0.0, shap)
    expected = shap + 1.0 / 4
    np.testing.assert_allclose(out, expected)


def test_change_shap_base_value_3d() -> None:
    """3-D interaction cube shifts main and interaction effects."""
    cube = np.zeros((1, 3, 3))
    out = _change_shap_base_value(2.0, 0.0, cube)
    assert out.shape == cube.shape
    # Diagonal gets extra temp; off-diagonal get temp once
    assert out[0, 0, 0] != 0.0


# ---------------------------------------------------------------------------
# DecisionPlotResult
# ---------------------------------------------------------------------------


def test_decision_plot_result_attrs() -> None:
    """DecisionPlotResult stores all constructor arguments."""
    shap = _make_shap(1, 3)
    idx = np.arange(3)
    result = DecisionPlotResult(
        base_value=0.5,
        shap_values=shap,
        feature_names=["a", "b", "c"],
        feature_idx=idx,
        xlim=(-1.0, 1.0),
    )
    assert result.base_value == 0.5
    assert result.shap_values is shap
    assert result.feature_names == ["a", "b", "c"]
    np.testing.assert_array_equal(result.feature_idx, idx)
    assert result.xlim == (-1.0, 1.0)


# ---------------------------------------------------------------------------
# _decision_plot_matplotlib
# ---------------------------------------------------------------------------


def _call_mpl(
    *,
    n_obs: int = 2,
    n_feat: int = 4,
    features: NDArray[Any] | None = None,
    highlight: Any = None,  # noqa: ANN401
    color_bar: bool = True,
    auto_size_plot: bool = True,
    title: str | None = None,
    ascending: bool = False,
    legend_labels: list[str] | None = None,
    feature_names: list[str] | None = None,
) -> None:
    """Invoke _decision_plot_matplotlib with sensible defaults."""
    plt.figure()
    base = 0.0
    cumsum = np.cumsum(
        np.concatenate(
            [np.full((n_obs, 1), base), np.ones((n_obs, n_feat))],
            axis=1,
        ),
        axis=1,
    )
    names = feature_names or [f"f{i}" for i in range(n_feat)]
    cmap = LinearSegmentedColormap.from_list("rb", ["#ff0000", "#0000ff"])
    _decision_plot_matplotlib(
        base_value=base,
        cumsum=cumsum,
        ascending=ascending,
        feature_display_count=n_feat,
        features=features,
        feature_names=names,
        highlight=highlight,
        plot_color=cmap,
        axis_color="#333333",
        y_demarc_color="#333333",
        xlim=(-1.0, float(n_feat) + 1.0),
        alpha=1.0,
        color_bar=color_bar,
        auto_size_plot=auto_size_plot,
        title=title,
        show=False,
        legend_labels=legend_labels,
        legend_location="best",
    )


def test_decision_plot_matplotlib_basic() -> None:
    """Basic multi-observation plot without optional extras."""
    _call_mpl(color_bar=False, auto_size_plot=False)


def test_decision_plot_matplotlib_auto_size() -> None:
    """auto_size_plot=True resizes the current figure."""
    _call_mpl(auto_size_plot=True)


def test_decision_plot_matplotlib_highlight() -> None:
    """Highlight path uses dashed linestyle and thicker stroke."""
    _call_mpl(highlight=np.array([0]))


def test_decision_plot_matplotlib_interaction_fontsize() -> None:
    """Interaction feature names force smaller fontsize."""
    names = ["a", "b *\nc", "d", "e"]
    _call_mpl(n_feat=4, feature_names=names, color_bar=False)


def test_decision_plot_matplotlib_single_numeric_features() -> None:
    """Single observation with numeric feature values draws labels."""
    feats = np.array([[1.5, 2.0, 3.25, 4.0]])
    _call_mpl(n_obs=1, n_feat=4, features=feats, color_bar=False)


def test_decision_plot_matplotlib_single_string_features() -> None:
    """Single observation with string feature values draws labels."""
    feats = np.array([["cat", "dog", "bird", "fish"]], dtype=object)
    _call_mpl(n_obs=1, n_feat=4, features=feats, color_bar=False)


def test_decision_plot_matplotlib_label_overflow_right() -> None:
    """Label that overflows right edge is flipped to the left."""
    # Tight xlim so the default left-aligned label overflows.
    plt.figure()
    base = 0.0
    n_feat = 3
    cumsum = np.array([[0.0, 0.9, 0.95, 1.0]])
    feats = np.array([[0.1, 0.2, 0.3]])
    names = ["f0", "f1", "f2"]
    cmap = LinearSegmentedColormap.from_list("rb", ["#ff0000", "#0000ff"])
    _decision_plot_matplotlib(
        base_value=base,
        cumsum=cumsum,
        ascending=False,
        feature_display_count=n_feat,
        features=feats,
        feature_names=names,
        highlight=None,
        plot_color=cmap,
        axis_color="#333333",
        y_demarc_color="#333333",
        xlim=(0.0, 1.0),
        alpha=1.0,
        color_bar=False,
        auto_size_plot=False,
        title=None,
        show=False,
        legend_labels=None,
        legend_location="best",
    )


def test_decision_plot_matplotlib_label_overflow_both() -> None:
    """Label that overflows both edges falls back to left at xlim[0]."""
    plt.figure()
    base = 0.0
    n_feat = 2
    # Extremely narrow xlim forces both overflow branches.
    cumsum = np.array([[0.0, 0.5, 1.0]])
    feats = np.array([[123456789.123, 987654321.987]])
    names = ["very_long_feature_name_0", "very_long_feature_name_1"]
    cmap = LinearSegmentedColormap.from_list("rb", ["#ff0000", "#0000ff"])
    _decision_plot_matplotlib(
        base_value=base,
        cumsum=cumsum,
        ascending=False,
        feature_display_count=n_feat,
        features=feats,
        feature_names=names,
        highlight=None,
        plot_color=cmap,
        axis_color="#333333",
        y_demarc_color="#333333",
        xlim=(0.4, 0.6),
        alpha=1.0,
        color_bar=False,
        auto_size_plot=False,
        title=None,
        show=False,
        legend_labels=None,
        legend_location="best",
    )


def test_decision_plot_matplotlib_color_bar() -> None:
    """color_bar=True adds a horizontal colour bar."""
    _call_mpl(color_bar=True)


def test_decision_plot_matplotlib_title() -> None:
    """Title is rendered via plt.title."""
    _call_mpl(title="My Decision", color_bar=False)


def test_decision_plot_matplotlib_ascending() -> None:
    """ascending=True inverts the y-axis."""
    _call_mpl(ascending=True, color_bar=False)


def test_decision_plot_matplotlib_legend() -> None:
    """legend_labels draws a legend."""
    _call_mpl(
        legend_labels=["obs0", "obs1"],
        color_bar=False,
    )


# ---------------------------------------------------------------------------
# _shap_decision - validation & type branches
# ---------------------------------------------------------------------------


def test_shap_decision_base_value_1d_array() -> None:
    """Single-element ndarray base_value is converted to float."""
    shap = _make_shap(1, 3)
    result = _shap_decision(
        base_value=np.array([0.5]),
        shap_values=shap,
        show=False,
        return_objects=True,
        ignore_warnings=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert isinstance(result, DecisionPlotResult)


def test_shap_decision_multi_output_list_base() -> None:
    """List base_value raises TypeError (multi-output hint)."""
    with pytest.raises(TypeError, match="multi output"):
        _shap_decision(
            base_value=[0.1, 0.2],  # type: ignore[arg-type]
            shap_values=_make_shap(1, 3),
            show=False,
        )


def test_shap_decision_multi_output_list_shap() -> None:
    """List shap_values raises TypeError (multi-output hint)."""
    with pytest.raises(TypeError, match="multi output"):
        _shap_decision(
            base_value=0.0,
            shap_values=[_make_shap(1, 3)],  # type: ignore[arg-type]
            show=False,
        )


def test_shap_decision_wrong_shap_type() -> None:
    """Non-ndarray shap_values raises TypeError."""
    with pytest.raises(TypeError, match="wrong type"):
        _shap_decision(
            base_value=0.0,
            shap_values="not an array",  # type: ignore[arg-type]
            show=False,
        )


def test_shap_decision_1d_shap_reshape() -> None:
    """1-D shap_values is reshaped to (1, n_features)."""
    shap_1d = np.array([0.1, 0.2, 0.3, 0.4])
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap_1d,
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert isinstance(result, DecisionPlotResult)
    assert result.shap_values.shape[0] == 1


# ---------------------------------------------------------------------------
# _shap_decision - feature / name normalisation
# ---------------------------------------------------------------------------


def test_shap_decision_feature_names_ndarray() -> None:
    """feature_names as ndarray is converted to list."""
    shap = _make_shap(1, 3)
    names = np.array(["a", "b", "c"])
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        feature_names=names,
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    assert "a" in result.feature_names


def test_shap_decision_features_dataframe() -> None:
    """DataFrame features supply column names when names are absent."""
    shap = _make_shap(1, 3)
    df = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["x", "y", "z"])
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        features=df,
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    assert "x" in result.feature_names


def test_shap_decision_features_dataframe_with_names() -> None:
    """DataFrame features with explicit feature_names keeps the names."""
    shap = _make_shap(1, 3)
    df = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["x", "y", "z"])
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        features=df,
        feature_names=["a", "b", "c"],
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    assert "a" in result.feature_names


def test_shap_decision_features_series() -> None:
    """Series features supply index as names when names are absent."""
    shap = _make_shap(1, 3)
    series = pd.Series([1.0, 2.0, 3.0], index=["p", "q", "r"])
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        features=series,
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    assert "p" in result.feature_names


def test_shap_decision_features_series_with_names() -> None:
    """Series features with explicit names keep the names."""
    shap = _make_shap(1, 3)
    series = pd.Series([1.0, 2.0, 3.0], index=["p", "q", "r"])
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        features=series,
        feature_names=["a", "b", "c"],
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    assert "a" in result.feature_names


def test_shap_decision_features_list() -> None:
    """List features become names; features_arr is None."""
    shap = _make_shap(1, 3)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        features=["alpha", "beta", "gamma"],
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    assert "alpha" in result.feature_names


def test_shap_decision_features_list_with_names() -> None:
    """List features with explicit names keep the names."""
    shap = _make_shap(1, 3)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        features=["alpha", "beta", "gamma"],
        feature_names=["a", "b", "c"],
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    assert "a" in result.feature_names


def test_shap_decision_features_1d_array_as_names() -> None:
    """1-D features array used as names when names_list is None."""
    shap = _make_shap(1, 3)
    feats = np.array(["u", "v", "w"], dtype=object)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        features=feats,
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    assert "u" in result.feature_names


def test_shap_decision_features_none() -> None:
    """features=None yields default Feature-i names."""
    shap = _make_shap(1, 3)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        features=None,
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    assert any("Feature" in n for n in result.feature_names)


def test_shap_decision_features_2d_array() -> None:
    """2-D ndarray features are accepted as features_arr."""
    shap = _make_shap(1, 3)
    feats = np.array([[1.0, 2.0, 3.0]])
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        features=feats,
        feature_names=["a", "b", "c"],
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert isinstance(result, DecisionPlotResult)


def test_shap_decision_features_unsupported_type() -> None:
    """Features that resolve to a non-ndarray raise TypeError."""
    shap = _make_shap(1, 3)

    class _WeirdFeatures:
        """Object that reaches the else branch and is not an ndarray."""

        ndim = 2

    with (
        patch(
            "xwhy.plots.visualisation.decision.np.asarray",
            return_value="not-an-array",
        ),
        pytest.raises(TypeError, match="unsupported type"),
    ):
        _shap_decision(
            base_value=0.0,
            shap_values=shap,
            features=_WeirdFeatures(),
            feature_names=["a", "b", "c"],
            show=False,
        )


def test_shap_decision_features_1d_reshape() -> None:
    """1-D numeric features_arr is reshaped to (1, n)."""
    shap = _make_shap(1, 3)
    # Pass via the else branch: 2-D path with names already set, then
    # a 1-D array through np.asarray would not hit the names branch.
    # Use a plain 1-D float array with feature_names so names_list is set
    # and we go through the else -> reshape path.
    feats = np.array([1.0, 2.0, 3.0])
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        features=feats,
        feature_names=["a", "b", "c"],
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert isinstance(result, DecisionPlotResult)


def test_shap_decision_feature_names_length_mismatch() -> None:
    """Mismatched feature_names length raises ValueError."""
    shap = _make_shap(1, 3)
    with pytest.raises(ValueError, match="must include all features"):
        _shap_decision(
            base_value=0.0,
            shap_values=shap,
            feature_names=["only_one"],
            show=False,
        )


# ---------------------------------------------------------------------------
# _shap_decision - interaction cube
# ---------------------------------------------------------------------------


def test_shap_decision_interaction_cube() -> None:
    """3-D interaction SHAP is flattened with interaction labels."""
    cube = _make_interaction(1, 3)
    result = _shap_decision(
        base_value=0.0,
        shap_values=cube,
        feature_names=["a", "b", "c"],
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
        ignore_warnings=True,
        # Display only a few so plot stays small
        feature_display_range=slice(-1, -4, -1),
    )
    assert result is not None
    # Interaction names contain " *\n"
    assert any(" *\n" in n for n in result.feature_names)


# ---------------------------------------------------------------------------
# _shap_decision - feature_order branches
# ---------------------------------------------------------------------------


def test_shap_decision_feature_order_list() -> None:
    """feature_order as list of ints is accepted."""
    shap = _make_shap(1, 4)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        feature_order=[3, 1, 0, 2],
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    np.testing.assert_array_equal(result.feature_idx, [3, 1, 0, 2])


def test_shap_decision_feature_order_ndarray() -> None:
    """feature_order as ndarray is accepted."""
    shap = _make_shap(1, 4)
    idx = np.array([2, 0, 1, 3])
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        feature_order=idx,
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    np.testing.assert_array_equal(result.feature_idx, idx)


def test_shap_decision_feature_order_none() -> None:
    """feature_order=None keeps natural order."""
    shap = _make_shap(1, 3)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        feature_order=None,
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    np.testing.assert_array_equal(result.feature_idx, np.arange(3))


def test_shap_decision_feature_order_none_str() -> None:
    """feature_order='none' (case-insensitive) keeps natural order."""
    shap = _make_shap(1, 3)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        feature_order="None",
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    np.testing.assert_array_equal(result.feature_idx, np.arange(3))


def test_shap_decision_feature_order_importance() -> None:
    """feature_order='importance' sorts by absolute SHAP sum."""
    shap = _make_shap(2, 4)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        feature_order="importance",
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    assert result.feature_idx.shape == (4,)


def test_shap_decision_feature_order_hclust() -> None:
    """feature_order='hclust' uses hierarchical clustering."""
    shap = _make_shap(5, 4)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        feature_order="hclust",
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    assert set(result.feature_idx.tolist()) == {0, 1, 2, 3}


def test_shap_decision_feature_order_invalid() -> None:
    """Unrecognised feature_order string raises ValueError."""
    with pytest.raises(ValueError, match="feature_order arg requires"):
        _shap_decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            feature_order="bogus",
            show=False,
        )


def test_shap_decision_feature_order_bad_length() -> None:
    """feature_order list of wrong length raises ValueError."""
    with pytest.raises(ValueError, match="length must match"):
        _shap_decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            feature_order=[0, 1],
            show=False,
        )


def test_shap_decision_feature_order_non_integer() -> None:
    """feature_order array of non-integer dtype raises ValueError."""
    with pytest.raises(ValueError, match="data type must"):
        _shap_decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            feature_order=np.array([0.0, 1.0, 2.0]),
            show=False,
        )


# ---------------------------------------------------------------------------
# _shap_decision - feature_display_range
# ---------------------------------------------------------------------------


def test_shap_decision_display_range_default() -> None:
    """Default feature_display_range is slice(-1, -21, -1)."""
    shap = _make_shap(1, 5)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert isinstance(result, DecisionPlotResult)


def test_shap_decision_display_range_invalid_type() -> None:
    """Non-slice/range feature_display_range raises TypeError."""
    with pytest.raises(TypeError, match="slice or a range"):
        _shap_decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            feature_display_range=42,  # type: ignore[arg-type]
            show=False,
        )


def test_shap_decision_display_range_bad_step() -> None:
    """Step other than 1/-1/None raises ValueError."""
    with pytest.raises(ValueError, match="step of 1, -1, or None"):
        _shap_decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            feature_display_range=slice(0, 3, 2),
            show=False,
        )


def test_shap_decision_display_range_range_object() -> None:
    """Range object is converted to an equivalent slice."""
    shap = _make_shap(1, 5)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        feature_display_range=range(0, 3),
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert isinstance(result, DecisionPlotResult)


def test_shap_decision_display_range_range_negative() -> None:
    """Range with a negative start maps that bound to the c_min sentinel."""
    shap = _make_shap(1, 5)
    # start < 0 -> c_min; stop >= 0 stays as-is so display_count > 0.
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        feature_display_range=range(-2, 3),
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
        ignore_warnings=True,
    )
    assert isinstance(result, DecisionPlotResult)


def test_shap_decision_display_range_step_neg1() -> None:
    """Negative step sets ascending=False and reorders indices."""
    shap = _make_shap(1, 5)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        feature_display_range=slice(-1, -4, -1),
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert isinstance(result, DecisionPlotResult)


def test_shap_decision_display_range_start_nonzero() -> None:
    """start_i != 0 uses the partial-cumsum branch."""
    shap = _make_shap(1, 6)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        feature_order=None,
        feature_display_range=slice(2, 5),
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert isinstance(result, DecisionPlotResult)


# ---------------------------------------------------------------------------
# _shap_decision - new_base_value, link, xlim, alpha, warnings
# ---------------------------------------------------------------------------


def test_shap_decision_new_base_value() -> None:
    """new_base_value shifts SHAP values via _change_shap_base_value.

    ``DecisionPlotResult.base_value`` stores the base used in the plot
    (i.e. ``new_base_value`` when it is set).
    """
    shap = _make_shap(1, 3)
    result = _shap_decision(
        base_value=1.0,
        shap_values=shap,
        new_base_value=0.0,
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    assert result.base_value == 0.0


def test_shap_decision_logit_link() -> None:
    """Logit link maps values to probability space and sets xlim."""
    # Use small SHAP so cumsum stays in a reasonable log-odds range
    shap = np.array([[0.1, -0.1, 0.05]])
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        link="logit",
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    assert result.xlim == (-0.02, 1.02)


def test_shap_decision_logit_link_with_xlim() -> None:
    """Logit link with explicit xlim does not override it."""
    shap = np.array([[0.1, -0.1, 0.05]])
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        link="logit",
        xlim=(0.0, 1.0),
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    assert result.xlim == (0.0, 1.0)


def test_shap_decision_identity_xlim_left_heavy() -> None:
    """When left span > right span, xlim is symmetric about base."""
    # Negative-dominant SHAP so n_left > m_right
    shap = np.array([[-2.0, -1.0, -0.5]])
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    assert result.xlim[0] < 0 < result.xlim[1]


def test_shap_decision_identity_xlim_right_heavy() -> None:
    """When right span >= left span, xlim is symmetric about base."""
    shap = np.array([[2.0, 1.0, 0.5]])
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    assert result.xlim[0] < 0 < result.xlim[1]


def test_shap_decision_explicit_xlim() -> None:
    """Provided xlim is kept as-is."""
    shap = _make_shap(1, 3)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        xlim=(-5.0, 5.0),
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is not None
    assert result.xlim == (-5.0, 5.0)


def test_shap_decision_alpha_none() -> None:
    """alpha=None defaults to 1.0."""
    shap = _make_shap(1, 3)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        alpha=None,
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert isinstance(result, DecisionPlotResult)


def test_shap_decision_alpha_and_plot_color_explicit() -> None:
    """Non-None alpha and plot_color skip the default-assignment branches."""
    shap = _make_shap(1, 3)
    cmap = LinearSegmentedColormap.from_list("rb", ["#ff0000", "#0000ff"])
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        alpha=0.5,
        plot_color=cmap,
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert isinstance(result, DecisionPlotResult)


def test_shap_decision_plot_color_none() -> None:
    """plot_color=None falls back to RED_BLUE."""
    shap = _make_shap(1, 3)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        plot_color=None,
        show=False,
        return_objects=True,
        color_bar=False,
        auto_size_plot=False,
    )
    assert isinstance(result, DecisionPlotResult)


def test_shap_decision_return_none() -> None:
    """return_objects=False returns None."""
    shap = _make_shap(1, 3)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        show=False,
        return_objects=False,
        color_bar=False,
        auto_size_plot=False,
    )
    assert result is None


def test_shap_decision_warn_too_many_observations() -> None:
    """observation_count > 2000 raises RuntimeError unless ignored."""
    shap = np.zeros((2001, 2))
    with pytest.raises(RuntimeError, match="observations may be slow"):
        _shap_decision(
            base_value=0.0,
            shap_values=shap,
            show=False,
            ignore_warnings=False,
            color_bar=False,
            auto_size_plot=False,
            feature_display_range=slice(0, 2),
        )


def test_shap_decision_warn_too_many_features() -> None:
    """feature_display_count > 200 raises RuntimeError unless ignored."""
    shap = np.zeros((1, 250))
    with pytest.raises(RuntimeError, match="features may create"):
        _shap_decision(
            base_value=0.0,
            shap_values=shap,
            show=False,
            ignore_warnings=False,
            color_bar=False,
            auto_size_plot=False,
            feature_display_range=slice(0, 250),
        )


def test_shap_decision_warn_product_too_large() -> None:
    """feature_count * observation_count > 1e8 raises RuntimeError.

    observation_count=2000 (not > 2000) and feature_display_count=2
    (not > 200) so only the product guard fires. float16 keeps the
    allocation around 190 MiB; plotting is mocked.
    """
    # 2000 * 50_001 = 100_002_000 > 100_000_000.
    shap = np.zeros((2000, 50_001), dtype=np.float16)
    with (
        patch(
            "xwhy.plots.visualisation.decision._decision_plot_matplotlib",
        ),
        pytest.raises(RuntimeError, match="Processing SHAP values"),
    ):
        _shap_decision(
            base_value=0.0,
            shap_values=shap,
            show=False,
            ignore_warnings=False,
            color_bar=False,
            auto_size_plot=False,
            feature_order=None,
            feature_display_range=slice(0, 2),
        )


def test_shap_decision_ignore_warnings() -> None:
    """ignore_warnings=True suppresses size checks."""
    # Non-zero values so auto-computed xlim is not singular (0, 0).
    shap = np.full((2001, 2), 0.1)
    result = _shap_decision(
        base_value=0.0,
        shap_values=shap,
        show=False,
        return_objects=True,
        ignore_warnings=True,
        color_bar=False,
        auto_size_plot=False,
        feature_display_range=slice(0, 2),
    )
    assert isinstance(result, DecisionPlotResult)


# ---------------------------------------------------------------------------
# decision - public API
# ---------------------------------------------------------------------------


def test_decision_matplotlib_basic() -> None:
    """Public decision() with matplotlib backend and show=False."""
    with patch(
        "xwhy.plots.visualisation.decision._finish_matplotlib",
        return_value="finished",
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(2, 4),
            show=False,
            title="Test",
            figsize=(8, 6),
        )
        assert result == "finished"


def test_decision_matplotlib_no_finish_show_true() -> None:
    """show=True without _finish_matplotlib calls plt.show()."""
    with (
        patch(
            "xwhy.plots.visualisation.decision._finish_matplotlib",
            new=None,
        ),
        patch("matplotlib.pyplot.show") as mock_show,
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            show=True,
        )
        assert result is None
        mock_show.assert_called_once()


def test_decision_matplotlib_no_finish_show_false() -> None:
    """show=False without _finish_matplotlib returns Figure."""
    with patch(
        "xwhy.plots.visualisation.decision._finish_matplotlib",
        new=None,
    ):
        fig = decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            show=False,
        )
        assert fig is not None


def test_decision_no_figsize() -> None:
    """figsize=None uses default plt.subplots()."""
    with patch(
        "xwhy.plots.visualisation.decision._finish_matplotlib",
        return_value="ok",
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            show=False,
            figsize=None,
        )
        assert result == "ok"


def test_decision_no_title() -> None:
    """title=None skips fig.suptitle."""
    with patch(
        "xwhy.plots.visualisation.decision._finish_matplotlib",
        return_value="ok",
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            show=False,
            title=None,
        )
        assert result == "ok"


def test_decision_base_value_array() -> None:
    """Ndarray base_value is reduced to a scalar."""
    with patch(
        "xwhy.plots.visualisation.decision._finish_matplotlib",
        return_value="ok",
    ):
        result = decision(
            base_value=np.array([0.5]),
            shap_values=_make_shap(1, 3),
            show=False,
        )
        assert result == "ok"


def test_decision_with_features_single() -> None:
    """Single-instance features are formatted into labels."""
    with patch(
        "xwhy.plots.visualisation.decision._finish_matplotlib",
        return_value="ok",
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            features=np.array([1.0, 2.0, 3.0]),
            feature_names=["a", "b", "c"],
            show=False,
        )
        assert result == "ok"


def test_decision_with_features_multi() -> None:
    """Multi-instance features keep plain feature names."""
    with patch(
        "xwhy.plots.visualisation.decision._finish_matplotlib",
        return_value="ok",
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(3, 3),
            features=np.ones((3, 3)),
            feature_names=["a", "b", "c"],
            show=False,
        )
        assert result == "ok"


def test_decision_features_wrong_shape() -> None:
    """Features with wrong column count are discarded."""
    with patch(
        "xwhy.plots.visualisation.decision._finish_matplotlib",
        return_value="ok",
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            features=np.array([[1.0, 2.0]]),  # wrong width
            show=False,
        )
        assert result == "ok"


def test_decision_max_display_truncates() -> None:
    """max_display limits the number of feature rows."""
    with patch(
        "xwhy.plots.visualisation.decision._finish_matplotlib",
        return_value="ok",
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(1, 10),
            max_display=3,
            show=False,
        )
        assert result == "ok"


def test_decision_max_display_none() -> None:
    """max_display=None shows all features."""
    with patch(
        "xwhy.plots.visualisation.decision._finish_matplotlib",
        return_value="ok",
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(1, 5),
            max_display=None,
            show=False,
        )
        assert result == "ok"


def test_decision_resolve_names_none() -> None:
    """When _resolve_names is None, fall back to local name logic."""
    with (
        patch(
            "xwhy.plots.visualisation.decision._resolve_names",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.decision._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            feature_names=["x", "y", "z"],
            show=False,
        )
        assert result == "ok"


def test_decision_resolve_names_none_no_names() -> None:
    """_resolve_names is None and no feature_names -> default names."""
    with (
        patch(
            "xwhy.plots.visualisation.decision._resolve_names",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.decision._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            feature_names=None,
            show=False,
        )
        assert result == "ok"


def test_decision_format_value_none() -> None:
    """_format_value is None falls back to str for feature labels."""
    with (
        patch(
            "xwhy.plots.visualisation.decision._format_value",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.decision._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            features=np.array([1.0, 2.0, 3.0]),
            feature_names=["a", "b", "c"],
            show=False,
        )
        assert result == "ok"


def test_decision_check_backend_none_valid() -> None:
    """_check_backend is None; valid backend is accepted."""
    with (
        patch(
            "xwhy.plots.visualisation.decision._check_backend",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.decision._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            backend="matplotlib",
            show=False,
        )
        assert result == "ok"


def test_decision_check_backend_none_invalid() -> None:
    """_check_backend is None; invalid backend raises ValueError."""
    with (
        patch(
            "xwhy.plots.visualisation.decision._check_backend",
            new=None,
        ),
        pytest.raises(ValueError, match="Unsupported backend"),
    ):
        decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            backend="invalid",
            show=False,
        )


def test_decision_plotly_backend() -> None:
    """Plotly backend builds a figure and delegates to _finish_plotly."""
    mock_finish = MagicMock(return_value="plotly_done")
    with (
        patch(
            "xwhy.plots.visualisation.decision._check_backend",
            return_value="plotly",
        ),
        patch(
            "xwhy.plots.visualisation.decision._finish_plotly",
            mock_finish,
        ),
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(2, 4),
            backend="plotly",
            show=False,
            title="Plotly Decision",
        )
        assert result == "plotly_done"
        mock_finish.assert_called_once()


def test_decision_plotly_no_finish_show_true() -> None:
    """Plotly, show=True, no _finish_plotly -> returns None."""
    with (
        patch(
            "xwhy.plots.visualisation.decision._check_backend",
            return_value="plotly",
        ),
        patch(
            "xwhy.plots.visualisation.decision._finish_plotly",
            new=None,
        ),
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            backend="plotly",
            show=True,
        )
        assert result is None


def test_decision_plotly_no_finish_show_false() -> None:
    """Plotly, show=False, no _finish_plotly -> returns figure."""
    with (
        patch(
            "xwhy.plots.visualisation.decision._check_backend",
            return_value="plotly",
        ),
        patch(
            "xwhy.plots.visualisation.decision._finish_plotly",
            new=None,
        ),
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            backend="plotly",
            show=False,
        )
        assert result is not None


def test_decision_plotly_red_blue_not_callable() -> None:
    """When RED_BLUE is not callable, a fixed rgb colour is used."""
    with (
        patch(
            "xwhy.plots.visualisation.decision._check_backend",
            return_value="plotly",
        ),
        patch(
            "xwhy.plots.visualisation.decision.RED_BLUE",
            new="not_callable",
        ),
        patch(
            "xwhy.plots.visualisation.decision._finish_plotly",
            return_value="ok",
        ),
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            backend="plotly",
            show=False,
        )
        assert result == "ok"


def test_decision_plotly_go_none() -> None:
    """Go is None raises ImportError for plotly backend."""
    with (
        patch(
            "xwhy.plots.visualisation.decision._check_backend",
            return_value="plotly",
        ),
        patch(
            "xwhy.plots.visualisation.decision.go",
            new=None,
        ),
        pytest.raises(ImportError, match="plotly is required"),
    ):
        decision(
            base_value=0.0,
            shap_values=_make_shap(1, 3),
            backend="plotly",
            show=False,
        )


def test_decision_predictions_equal_vmin_vmax() -> None:
    """When all predictions are equal, norm uses vmin + 1e-9."""
    # Identical rows -> identical predictions
    shap = np.zeros((2, 3))
    with (
        patch(
            "xwhy.plots.visualisation.decision._check_backend",
            return_value="plotly",
        ),
        patch(
            "xwhy.plots.visualisation.decision._finish_plotly",
            return_value="ok",
        ),
    ):
        result = decision(
            base_value=0.0,
            shap_values=shap,
            backend="plotly",
            show=False,
        )
        assert result == "ok"


def test_decision_kwargs_forwarded() -> None:
    """Extra kwargs are forwarded to _shap_decision."""
    with patch(
        "xwhy.plots.visualisation.decision._finish_matplotlib",
        return_value="ok",
    ):
        result = decision(
            base_value=0.0,
            shap_values=_make_shap(1, 4),
            show=False,
            feature_order="importance",
            link="identity",
            color_bar=False,
            ignore_warnings=True,
            new_base_value=None,
            legend_labels=None,
            legend_location="upper right",
            axis_color="#000000",
            y_demarc_color="#111111",
        )
        assert result == "ok"
