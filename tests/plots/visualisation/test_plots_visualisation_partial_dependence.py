"""Test partial_dependence module."""

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

from xwhy.plots.visualisation.partial_dependence import (
    _shap_partial_dependence,
    compute_bounds,
    partial_dependence,
)


@pytest.fixture(autouse=True)
def _close_plots() -> Any:  # noqa: ANN401
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(
    n_samples: int = 40,
    n_features: int = 3,
    seed: int = 0,
) -> NDArray[np.floating[Any]]:
    """Return a small deterministic feature matrix."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_samples, n_features)).astype(float)


def _linear_model(x: NDArray[Any]) -> NDArray[Any]:
    """Return the row-wise sum of a feature matrix."""
    return np.asarray(x, dtype=float).sum(axis=1)


def _patch_convert_name() -> Any:  # noqa: ANN401
    """Patch convert_name to resolve int/str indices simply."""

    def _convert(
        ind: int | str,
        _shap: object,
        names: list[str] | None,
    ) -> int:
        if isinstance(ind, int):
            return ind
        if names is None:
            msg = f"Unknown feature {ind!r}"
            raise ValueError(msg)
        return list(names).index(str(ind))

    return patch(
        "xwhy.plots.visualisation.partial_dependence.convert_name",
        side_effect=_convert,
    )


# ---------------------------------------------------------------------------
# compute_bounds
# ---------------------------------------------------------------------------


def test_compute_bounds_both_none() -> None:
    """When both bounds are None the inputs are returned unchanged."""
    xv = np.array([1.0, 2.0, 3.0])
    assert compute_bounds(None, None, xv) == (None, None)


def test_compute_bounds_percentile_strings() -> None:
    """percentile(x) strings are resolved against the feature values."""
    xv = np.linspace(0.0, 100.0, 101)
    xmin, xmax = compute_bounds("percentile(10)", "percentile(90)", xv)
    assert xmin is not None
    assert xmax is not None
    assert xmin < xmax


def test_compute_bounds_float_and_pad() -> None:
    """Matching nanmin/nanmax triggers padding on that side."""
    xv = np.array([0.0, 5.0, 10.0])
    xmin, xmax = compute_bounds(0.0, 10.0, xv)
    assert xmin is not None
    assert xmin < 0.0
    assert xmax is not None
    assert xmax > 10.0


def test_compute_bounds_partial_none() -> None:
    """One-sided None is padded relative to the resolved other bound."""
    xv = np.array([1.0, 2.0, 3.0])
    xmin, xmax = compute_bounds(None, 3.0, xv)
    assert xmin is not None
    assert xmax is not None


# ---------------------------------------------------------------------------
# _shap_partial_dependence — 1-D
# ---------------------------------------------------------------------------


def test_shap_pd_1d_basic() -> None:
    """1-D PD with ICE and histogram on a NumPy matrix."""
    data = _make_data()
    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=0,
            model=_linear_model,
            data=data,
            npoints=10,
            ice=True,
            hist=True,
            show=False,
        )
    assert result is not None
    fig, ax1 = result
    assert isinstance(fig, Figure)
    assert ax1 is not None


def test_shap_pd_1d_no_ice() -> None:
    """ice=False uses the E[f(x)|feature] ylabel branch."""
    data = _make_data()
    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=1,
            model=_linear_model,
            data=data,
            npoints=8,
            ice=False,
            hist=False,
            ylabel=None,
            show=False,
        )
    assert result is not None


def test_shap_pd_1d_custom_ylabel() -> None:
    """Explicit ylabel is applied without override."""
    data = _make_data()
    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=0,
            model=_linear_model,
            data=data,
            npoints=8,
            ylabel="custom y",
            show=False,
        )
    assert result is not None


def test_shap_pd_1d_dataframe() -> None:
    """DataFrame input supplies column names and uses DataFrame model calls."""
    data = pd.DataFrame(_make_data(), columns=["a", "b", "c"])

    def _df_model(df: pd.DataFrame) -> NDArray[Any]:
        return df.values.sum(axis=1)  # type: ignore[no-any-return]

    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind="a",
            model=_df_model,
            data=data,
            feature_names=None,
            npoints=8,
            ice=True,
            show=False,
        )
    assert result is not None


def test_shap_pd_1d_dataframe_with_names() -> None:
    """DataFrame with explicit feature_names keeps those names."""
    data = pd.DataFrame(_make_data(n_features=2), columns=["x", "y"])

    def _df_model(df: pd.DataFrame) -> NDArray[Any]:
        return df.values.sum(axis=1)  # type: ignore[no-any-return]

    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=0,
            model=_df_model,
            data=data,
            feature_names=["x", "y"],
            npoints=6,
            ice=False,
            show=False,
        )
    assert result is not None


def test_shap_pd_1d_explanation_type() -> None:
    """Explanation-named object supplies data and shap_values.

    ``base_values`` must be a scalar: matplotlib ``stem`` rejects an
    array-valued ``bottom``.
    """
    matrix = _make_data()
    shap_vals = np.random.default_rng(0).normal(size=matrix.shape)

    class Explanation:
        def __init__(self) -> None:
            self.data = matrix
            self.values = shap_vals
            self.base_values = 0.5

    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=0,
            model=_linear_model,
            data=Explanation(),
            npoints=8,
            ice=False,
            hist=True,
            show=False,
        )
    assert result is not None


def test_shap_pd_1d_with_ax() -> None:
    """Passing ax uses gcf/gca instead of creating a new figure."""
    data = _make_data()
    plt.figure()
    ax = plt.gca()
    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=0,
            model=_linear_model,
            data=data,
            npoints=6,
            ax=ax,
            show=False,
        )
    assert result is not None


def test_shap_pd_1d_ace_linewidth_auto() -> None:
    """ace_linewidth='auto' scales line width by sample count."""
    data = _make_data(n_samples=80)
    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=0,
            model=_linear_model,
            data=data,
            npoints=6,
            ice=True,
            ace_linewidth="auto",
            show=False,
        )
    assert result is not None


def test_shap_pd_1d_ace_linewidth_float() -> None:
    """Explicit ace_linewidth float is used for ICE lines."""
    data = _make_data()
    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=0,
            model=_linear_model,
            data=data,
            npoints=6,
            ice=True,
            ace_linewidth=0.5,
            show=False,
        )
    assert result is not None


def test_shap_pd_1d_feature_expected_value() -> None:
    """feature_expected_value draws a vertical E[feature] line."""
    data = _make_data()
    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=0,
            model=_linear_model,
            data=data,
            npoints=6,
            feature_expected_value=True,
            show=False,
        )
    assert result is not None


def test_shap_pd_1d_model_expected_true() -> None:
    """model_expected_value=True computes E[f(x)] from the model."""
    data = _make_data()
    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=0,
            model=_linear_model,
            data=data,
            npoints=6,
            model_expected_value=True,
            show=False,
        )
    assert result is not None


def test_shap_pd_1d_model_expected_true_dataframe() -> None:
    """model_expected_value=True with DataFrame uses DataFrame model call."""
    data = pd.DataFrame(_make_data(), columns=["a", "b", "c"])

    def _df_model(df: pd.DataFrame) -> NDArray[Any]:
        return df.values.sum(axis=1)  # type: ignore[no-any-return]

    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=0,
            model=_df_model,
            data=data,
            npoints=6,
            model_expected_value=True,
            ice=False,
            show=False,
        )
    assert result is not None


def test_shap_pd_1d_model_expected_float() -> None:
    """model_expected_value as a float uses that fixed value."""
    data = _make_data()
    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=0,
            model=_linear_model,
            data=data,
            npoints=6,
            model_expected_value=0.5,
            show=False,
        )
    assert result is not None


def test_shap_pd_1d_shap_values_stems() -> None:
    """shap_values draws stem markers from base_values + values.

    ``base_values`` is a scalar so matplotlib ``stem`` accepts ``bottom``.
    """
    data = _make_data()
    shap = SimpleNamespace(
        data=data,
        values=np.random.default_rng(0).normal(size=data.shape),
        base_values=0.1,
    )
    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=0,
            model=_linear_model,
            data=data,
            npoints=6,
            shap_values=shap,
            model_expected_value=False,
            ice=False,
            show=False,
        )
    assert result is not None


def test_shap_pd_1d_shap_values_mev_from_base() -> None:
    """Non-numeric model_expected_value uses shap_values.base_values (line 248).

    ``False`` is a subclass of ``int``, so it would take the float branch.
    Passing ``None`` forces the else path that reads ``base_values``.
    """
    data = _make_data()
    shap = SimpleNamespace(
        data=data,
        values=np.random.default_rng(1).normal(size=data.shape),
        base_values=0.25,
    )
    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=0,
            model=_linear_model,
            data=data,
            npoints=6,
            shap_values=shap,
            model_expected_value=None,  # type: ignore[arg-type]
            show=False,
        )
    assert result is not None


def test_shap_pd_1d_show_true() -> None:
    """show=True calls plt.show() and returns None."""
    data = _make_data()
    with (
        _patch_convert_name(),
        patch("matplotlib.pyplot.show") as mock_show,
    ):
        result = _shap_partial_dependence(
            ind=0,
            model=_linear_model,
            data=data,
            npoints=6,
            ice=False,
            show=True,
        )
        assert result is None
        mock_show.assert_called_once()


def test_shap_pd_1d_default_feature_names() -> None:
    """feature_names=None generates Feature-i labels."""
    data = _make_data()
    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=0,
            model=_linear_model,
            data=data,
            feature_names=None,
            npoints=6,
            ice=False,
            show=False,
        )
    assert result is not None


def test_shap_pd_1d_npoints_none() -> None:
    """npoints=None defaults to 100 for 1-D."""
    data = _make_data(n_samples=20)
    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=0,
            model=_linear_model,
            data=data,
            npoints=None,
            ice=False,
            hist=False,
            show=False,
        )
    assert result is not None


# ---------------------------------------------------------------------------
# _shap_partial_dependence — 2-D
# ---------------------------------------------------------------------------


def test_shap_pd_2d_basic() -> None:
    """2-D surface plot for a pair of feature indices."""
    data = _make_data(n_samples=30, n_features=3)
    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=(0, 1),
            model=_linear_model,
            data=data,
            npoints=5,
            show=False,
        )
    assert result is not None
    fig, _ax = result
    assert isinstance(fig, Figure)


def test_shap_pd_2d_tuple_bounds() -> None:
    """xmin/xmax tuples supply per-axis bounds for 2-D."""
    data = _make_data(n_samples=30, n_features=3)
    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=(0, 1),
            model=_linear_model,
            data=data,
            npoints=4,
            xmin=("percentile(5)", "percentile(5)"),
            xmax=("percentile(95)", "percentile(95)"),
            show=False,
        )
    assert result is not None


def test_shap_pd_2d_scalar_bounds() -> None:
    """Scalar xmin/xmax are shared across both axes."""
    data = _make_data(n_samples=30, n_features=3)
    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=(0, 1),
            model=_linear_model,
            data=data,
            npoints=4,
            xmin="percentile(0)",
            xmax="percentile(100)",
            show=False,
        )
    assert result is not None


def test_shap_pd_2d_show_true() -> None:
    """2-D show=True returns None."""
    data = _make_data(n_samples=20, n_features=2)
    with (
        _patch_convert_name(),
        patch("matplotlib.pyplot.show") as mock_show,
    ):
        result = _shap_partial_dependence(
            ind=(0, 1),
            model=_linear_model,
            data=data,
            npoints=4,
            show=True,
        )
        assert result is None
        mock_show.assert_called_once()


def test_shap_pd_2d_npoints_none() -> None:
    """npoints=None defaults to 20 for 2-D."""
    data = _make_data(n_samples=20, n_features=2)
    with _patch_convert_name():
        result = _shap_partial_dependence(
            ind=(0, 1),
            model=_linear_model,
            data=data,
            npoints=None,
            show=False,
        )
    assert result is not None


# ---------------------------------------------------------------------------
# partial_dependence (public)
# ---------------------------------------------------------------------------


def test_pd_with_finish() -> None:
    """When _finish_matplotlib is available, delegate to it."""
    data = _make_data()
    with (
        _patch_convert_name(),
        patch(
            "xwhy.plots.visualisation.partial_dependence._finish_matplotlib",
            return_value="finished",
        ),
    ):
        result = partial_dependence(
            0,
            _linear_model,
            data,
            npoints=6,
            ice=False,
            show=True,
            title="PD",
            figsize=(6, 4),
        )
        assert result == "finished"  # type: ignore[comparison-overlap]


def test_pd_show_true_no_finish() -> None:
    """show=True without _finish_matplotlib calls plt.show()."""
    data = _make_data()
    with (
        _patch_convert_name(),
        patch(
            "xwhy.plots.visualisation.partial_dependence._finish_matplotlib",
            new=None,
        ),
        patch("matplotlib.pyplot.show") as mock_show,
    ):
        result = partial_dependence(
            0, _linear_model, data, npoints=6, ice=False, show=True
        )
        assert result is None
        mock_show.assert_called_once()


def test_pd_show_false_no_finish() -> None:
    """show=False without _finish_matplotlib returns Figure."""
    data = _make_data()
    with (
        _patch_convert_name(),
        patch(
            "xwhy.plots.visualisation.partial_dependence._finish_matplotlib",
            new=None,
        ),
    ):
        fig = partial_dependence(
            0, _linear_model, data, npoints=6, ice=False, show=False
        )
        assert fig is not None
        assert isinstance(fig, Figure)


def test_pd_no_figsize_no_title() -> None:
    """figsize=None and title=None skip those branches."""
    data = _make_data()
    with (
        _patch_convert_name(),
        patch(
            "xwhy.plots.visualisation.partial_dependence._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = partial_dependence(
            0,
            _linear_model,
            data,
            npoints=6,
            ice=False,
            show=False,
            figsize=None,
            title=None,
        )
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_pd_kwargs_forwarded() -> None:
    """hist, expected-value flags, opacities, and bounds reach the helper."""
    data = _make_data()
    with (
        _patch_convert_name(),
        patch(
            "xwhy.plots.visualisation.partial_dependence._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = partial_dependence(
            0,
            _linear_model,
            data,
            npoints=6,
            ice=True,
            show=False,
            hist=False,
            model_expected_value=True,
            feature_expected_value=True,
            ylabel="y",
            ace_opacity=0.5,
            pd_opacity=0.8,
            pd_linewidth=1.5,
            ace_linewidth=0.3,
            xmin="percentile(5)",
            xmax="percentile(95)",
        )
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_pd_2d_via_public() -> None:
    """Public API accepts a feature-index pair for a 2-D surface."""
    data = _make_data(n_samples=25, n_features=3)
    with (
        _patch_convert_name(),
        patch(
            "xwhy.plots.visualisation.partial_dependence._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = partial_dependence(
            (0, 1),
            _linear_model,
            data,
            npoints=4,
            ice=False,
            show=False,
        )
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_pd_check_backend_none_valid() -> None:
    """_check_backend is None; valid backend is accepted."""
    data = _make_data()
    with (
        patch(
            "xwhy.plots.visualisation.partial_dependence._check_backend",
            new=None,
        ),
        _patch_convert_name(),
        patch(
            "xwhy.plots.visualisation.partial_dependence._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = partial_dependence(
            0,
            _linear_model,
            data,
            npoints=6,
            ice=False,
            backend="matplotlib",
            show=False,
        )
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_pd_check_backend_none_invalid() -> None:
    """_check_backend is None; invalid backend raises ValueError."""
    data = _make_data()
    with (
        patch(
            "xwhy.plots.visualisation.partial_dependence._check_backend",
            new=None,
        ),
        pytest.raises(ValueError, match="Unsupported backend"),
    ):
        partial_dependence(0, _linear_model, data, backend="plotly", show=False)


def test_pd_check_backend_called() -> None:
    """When _check_backend is present it is invoked."""
    data = _make_data()
    mock_check = MagicMock()
    with (
        patch(
            "xwhy.plots.visualisation.partial_dependence._check_backend",
            mock_check,
        ),
        _patch_convert_name(),
        patch(
            "xwhy.plots.visualisation.partial_dependence._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = partial_dependence(
            0, _linear_model, data, npoints=6, ice=False, show=False
        )
        assert result == "ok"  # type: ignore[comparison-overlap]
        mock_check.assert_called_once()
