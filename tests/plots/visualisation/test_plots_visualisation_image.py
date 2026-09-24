"""Test image module."""

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
import scipy.sparse
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from numpy.typing import NDArray

from xwhy.plots.visualisation.image import (
    DenseData,
    _build_output_tokens_html,
    _escape_token,
    _flatten_output_names,
    _grayscale_rgba,
    _image_viz_html,
    _image_viz_script,
    _shap_color_maps,
    _shap_image,
    _shap_image_to_text,
    _to_display_image,
    image,
    image_to_text,
    kmeans,
    ordinal_str,
)


@pytest.fixture(autouse=True)
def _close_plots() -> Any:  # noqa: ANN401
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rgb_batch(
    n: int = 1,
    h: int = 8,
    w: int = 8,
    seed: int = 0,
) -> NDArray[np.floating[Any]]:
    """Return a batch of random RGB images in [0, 1].

    Shape is always ``(n, H, W, 3)`` so ``_shap_image`` can index samples.
    """
    rng = np.random.default_rng(seed)
    return rng.random((n, h, w, 3)).astype(float)


def _shap_batch(
    n: int = 1,
    h: int = 8,
    w: int = 8,
    seed: int = 1,
) -> NDArray[np.floating[Any]]:
    """Return a batch of SHAP maps with a trailing channel axis.

    Shape is ``(n, H, W, 1)``. A 4-D layout avoids the 3-D "single image"
    reshape branch that would otherwise mangle the sample axis.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, h, w, 1)).astype(float)


def _make_explanation(
    n: int = 1,
    h: int = 6,
    w: int = 6,
    *,
    seed: int = 0,
) -> Any:  # noqa: ANN401
    """Build an object whose type name is ``Explanation``.

    ``_shap_image`` dispatches on ``type(...).__name__ == "Explanation"``.
    """

    class Explanation:
        def __init__(self) -> None:
            self.values = _shap_batch(n, h, w, seed=seed)
            self.data = _rgb_batch(n, h, w, seed=seed + 1)
            self.output_names = ["c0"]

    return Explanation()


# ---------------------------------------------------------------------------
# DenseData
# ---------------------------------------------------------------------------


def test_dense_data_default_groups() -> None:
    """Default groups are one index per feature name."""
    data = np.ones((5, 3))
    dd = DenseData(data, ["a", "b", "c"])
    assert dd.groups_size == 3
    assert not dd.transposed
    assert len(dd.weights) == 5


def test_dense_data_custom_groups() -> None:
    """Explicit groups list is stored."""
    data = np.ones((4, 4))
    groups = [np.array([0, 1]), np.array([2, 3])]
    dd = DenseData(data, ["g0", "g1"], groups)
    assert dd.groups_size == 2
    assert len(dd.groups[0]) == 2


def test_dense_data_transposed() -> None:
    """Data is detected as transposed when groups match axis 0."""
    # 3 features, data stored as (features, samples)
    data = np.ones((3, 10))
    dd = DenseData(data, ["a", "b", "c"])
    assert dd.transposed
    assert len(dd.weights) == 10


def test_dense_data_invalid_shape() -> None:
    """Mismatched name count raises ValueError."""
    data = np.ones((5, 4))
    with pytest.raises(ValueError, match="names must match"):
        DenseData(data, ["a", "b"])


def test_dense_data_custom_weights() -> None:
    """Per-sample weights are normalised."""
    data = np.ones((3, 2))
    weights = np.array([1.0, 2.0, 1.0])
    dd = DenseData(data, ["a", "b"], None, weights)
    assert dd.weights.sum() == pytest.approx(1.0)


def test_dense_data_invalid_weights() -> None:
    """Wrong weight length raises ValueError."""
    data = np.ones((3, 2))
    with pytest.raises(ValueError, match="weights must match"):
        DenseData(data, ["a", "b"], None, np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# kmeans
# ---------------------------------------------------------------------------


def test_kmeans_numpy() -> None:
    """Kmeans on a dense NumPy array returns DenseData."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 4))
    result = kmeans(x, k=3, round_values=True)
    assert isinstance(result, DenseData)
    assert result.data.shape == (3, 4)


def test_kmeans_dataframe() -> None:
    """Kmeans on a DataFrame uses column names as group_names."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(20, 3)), columns=["x", "y", "z"])
    result = kmeans(df, k=2, round_values=False)
    assert result.group_names == ["x", "y", "z"]
    assert result.data.shape == (2, 3)


def test_kmeans_sparse_round() -> None:
    """Kmeans with sparse input and round_values uses toarray path."""
    rng = np.random.default_rng(2)
    dense = np.abs(rng.normal(size=(15, 3)))
    sparse = scipy.sparse.csr_matrix(dense)
    result = kmeans(sparse, k=2, round_values=True)
    assert isinstance(result, DenseData)
    assert result.data.shape == (2, 3)


# ---------------------------------------------------------------------------
# _to_display_image
# ---------------------------------------------------------------------------


def test_to_display_image_rgb() -> None:
    """RGB image returns grayscale and original RGB."""
    rgb = np.random.default_rng(0).random((6, 6, 3))
    gray, display = _to_display_image(rgb)
    assert gray.shape == (6, 6)
    assert display.shape == (6, 6, 3)


def test_to_display_image_single_channel() -> None:
    """HxWx1 is reshaped to HxW."""
    mono = np.random.default_rng(0).random((5, 5, 1))
    gray, display = _to_display_image(mono)
    assert gray.shape == (5, 5)
    assert display.shape == (5, 5)


def test_to_display_image_grayscale_2d() -> None:
    """Already 2-D array is returned as both gray and display."""
    img = np.random.default_rng(0).random((7, 7))
    gray, display = _to_display_image(img)
    np.testing.assert_array_equal(gray, img)
    np.testing.assert_array_equal(display, img)


def test_to_display_image_multi_channel() -> None:
    """Non-RGB multi-channel projects onto 3 k-means centres."""
    img = np.random.default_rng(0).random((6, 6, 5))
    gray, display = _to_display_image(img)
    assert gray.shape == (6, 6)
    assert display.shape == (6, 6, 3)
    assert display.min() >= 0.0
    assert display.max() <= 1.0


# ---------------------------------------------------------------------------
# _shap_image
# ---------------------------------------------------------------------------


def test_shap_image_list_with_pixels() -> None:
    """List of SHAP arrays with pixel_values draws a figure."""
    pixels = _rgb_batch(2, 8, 8)
    shap = [_shap_batch(2, 8, 8)]
    fig = _shap_image(shap, pixel_values=pixels, show=False)
    assert isinstance(fig, Figure)


def test_shap_image_single_array() -> None:
    """Single non-list SHAP array is wrapped into a list."""
    pixels = _rgb_batch(1, 6, 6)
    shap = _shap_batch(1, 6, 6)
    fig = _shap_image(shap, pixel_values=pixels, show=False)
    assert isinstance(fig, Figure)


def test_shap_image_cmap_none() -> None:
    """cmap=None falls back to RED_TRANSPARENT_BLUE."""
    pixels = _rgb_batch(1, 6, 6)
    shap = [_shap_batch(1, 6, 6)]
    fig = _shap_image(shap, pixel_values=pixels, cmap=None, show=False)
    assert isinstance(fig, Figure)


def test_shap_image_explanation_2d() -> None:
    """Explanation-named object uses .data and .output_names."""
    exp = _make_explanation(1, 6, 6)
    fig = _shap_image(exp, pixel_values=None, labels=None, show=False)
    assert isinstance(fig, Figure)


def test_shap_image_explanation_labels_provided() -> None:
    """Explanation path with labels already set skips output_names (264→278)."""
    exp = _make_explanation(1, 6, 6)
    labels = np.array([["provided_label"]])
    fig = _shap_image(exp, pixel_values=None, labels=labels, show=False)
    assert isinstance(fig, Figure)


def test_shap_image_explanation_data_none_raises() -> None:
    """Explanation with data=None and no pixel_values raises ValueError (279-280)."""

    class Explanation:
        def __init__(self) -> None:
            self.values = _shap_batch(1, 4, 4)
            self.data = None
            self.output_names = ["c0"]

    with pytest.raises(ValueError, match="pixel_values is required"):
        _shap_image(Explanation(), pixel_values=None, show=False)


def test_shap_image_explanation_multi_output() -> None:
    """Explanation with values.ndim >= 5 splits into per-output list."""
    n, h, w, c, outs = 1, 4, 4, 1, 2
    values = np.random.default_rng(0).normal(size=(n, h, w, c, outs))
    pixels = np.random.default_rng(1).random((n, h, w, 3))

    class Explanation:
        def __init__(self) -> None:
            self.values = values
            self.data = pixels
            self.output_names = ["a", "b"]

    fig = _shap_image(Explanation(), show=False)
    assert isinstance(fig, Figure)


def test_shap_image_missing_pixels_raises() -> None:
    """Non-Explanation without pixel_values raises AssertionError."""
    shap = [_shap_batch(1, 4, 4)]
    with pytest.raises(AssertionError, match="pixel_values must be"):
        _shap_image(shap, pixel_values=None, show=False)


def test_shap_image_2d_maps_abs_vals() -> None:
    """2-D SHAP maps (no channel axis) take the abs_vals stack branch (319).

    ``(n, H, W)`` maps are 3-D, so the single-image reshape gate would fire.
    A thin ndarray subclass reports a 4-D shape for that gate while still
    returning 2-D ``(H, W)`` rows on ``[row]``, which is what the abs_vals
    branch keys on.
    """
    n, h, w = 2, 6, 6
    pixels = _rgb_batch(n, h, w)
    maps = np.random.default_rng(1).normal(size=(n, h, w))

    class _NoReshapeArray(np.ndarray):
        """Report 4-D shape so the ``len(shape)==3`` reshape is skipped."""

        def __new__(cls, input_array: NDArray[Any]) -> _NoReshapeArray:
            return np.asarray(input_array).view(cls)

        @property  # type: ignore[misc]
        def shape(self) -> tuple[int, ...]:
            base = self.view(np.ndarray)
            if base.ndim == 3:
                return (*base.shape, 1)
            return base.shape  # type: ignore[no-any-return]

        def __getitem__(self, key: object) -> Any:  # noqa: ANN401
            return self.view(np.ndarray)[key]  # type: ignore[call-overload]

    wrapped = _NoReshapeArray(maps)
    fig = _shap_image([wrapped], pixel_values=pixels, show=False)
    assert isinstance(fig, Figure)


def test_shap_image_im_none_skips_colorbar() -> None:
    """When no rows are drawn, ``im`` stays None and colorbar is skipped.

    Matplotlib rejects ``nrows=0``, so ``plt.subplots`` is patched to build a
    1-row figure while the real ``range(images.shape[0])`` loop still iterates
    zero times (354→366).
    """
    pixels = np.zeros((0, 4, 4, 3))
    shap = [np.zeros((0, 4, 4, 1))]

    real_subplots = plt.subplots

    def _safe_subplots(
        *args: object,
        **kwargs: object,
    ) -> tuple[Figure, Any]:
        kwargs = dict(kwargs)
        if kwargs.get("nrows") == 0:
            kwargs["nrows"] = 1
        return real_subplots(*args, **kwargs)  # type: ignore[no-any-return, call-overload]

    with patch(
        "xwhy.plots.visualisation.image.plt.subplots",
        side_effect=_safe_subplots,
    ):
        fig = _shap_image(shap, pixel_values=pixels, show=False)
    assert isinstance(fig, Figure)


def test_shap_image_3d_reshape() -> None:
    """3-D single-image SHAP (H, W, C) gets a batch dimension."""
    pixels = _rgb_batch(1, 5, 5)[0]  # HxWx3
    shap = np.random.default_rng(0).normal(size=(5, 5, 1))  # HxWx1
    fig = _shap_image(shap, pixel_values=pixels, show=False)
    assert isinstance(fig, Figure)


def test_shap_image_with_labels_and_true_labels() -> None:
    """Labels and true_labels set subplot titles."""
    pixels = _rgb_batch(2, 6, 6)
    shap = [_shap_batch(2, 6, 6)]
    labels_arr = np.array([["c0"], ["c1"]])
    fig = _shap_image(
        shap,
        pixel_values=pixels,
        labels=labels_arr,
        true_labels=["true0", "true1"],
        labelpad=4.0,
        show=False,
    )
    assert isinstance(fig, Figure)


def test_shap_image_vmax_and_hspace_auto() -> None:
    """Explicit vmax and hspace='auto' use tight_layout."""
    pixels = _rgb_batch(1, 6, 6)
    shap = [_shap_batch(1, 6, 6)]
    fig = _shap_image(
        shap,
        pixel_values=pixels,
        vmax=0.5,
        hspace="auto",
        show=False,
    )
    assert isinstance(fig, Figure)


def test_shap_image_multi_channel_shap() -> None:
    """SHAP with channel axis is summed over channels for overlay."""
    pixels = _rgb_batch(1, 6, 6)
    # (n, H, W, C)
    shap = [np.random.default_rng(0).normal(size=(1, 6, 6, 3))]
    fig = _shap_image(shap, pixel_values=pixels, show=False)
    assert isinstance(fig, Figure)


def test_shap_image_width_scales_down() -> None:
    """Wide layout is scaled when fig width exceeds *width*."""
    pixels = _rgb_batch(1, 4, 4)
    # Many outputs force large fig width
    shap = [_shap_batch(1, 4, 4) for _ in range(5)]
    fig = _shap_image(shap, pixel_values=pixels, width=8, show=False)
    assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# ordinal_str / _escape_token / _flatten_output_names
# ---------------------------------------------------------------------------


def test_ordinal_str() -> None:
    """Ordinal suffixes cover 1st, 2nd, 3rd, 4th, and teens."""
    assert ordinal_str(1) == "1st"
    assert ordinal_str(2) == "2nd"
    assert ordinal_str(3) == "3rd"
    assert ordinal_str(4) == "4th"
    assert ordinal_str(11) == "11th"
    assert ordinal_str(21) == "21st"
    assert ordinal_str(22) == "22nd"
    assert ordinal_str(23) == "23rd"


def test_escape_token() -> None:
    """HTML-sensitive characters and special markers are escaped."""
    assert "&lt;" in _escape_token("<tag>")
    assert "&gt;" in _escape_token("a>b")
    assert "##" not in _escape_token("hello ##world")
    assert "▁" not in _escape_token("▁token")
    assert "Ġ" not in _escape_token("Ġword")


def test_flatten_output_names_with_flatten() -> None:
    """Object with .flatten() is flattened to a list."""
    arr = np.array([["a", "b"], ["c", "d"]])
    result = _flatten_output_names(arr)
    assert result == ["a", "b", "c", "d"]


def test_flatten_output_names_nested_list() -> None:
    """Single nested list is unwrapped."""
    result = _flatten_output_names([["tok1", "tok2"]])
    assert result == ["tok1", "tok2"]


def test_flatten_output_names_plain_list() -> None:
    """Plain list falls through to the final ``list(model_output)`` branch."""
    result = _flatten_output_names(["x", "y", "z"])
    assert result == ["x", "y", "z"]


def test_flatten_output_names_nested_ndarray() -> None:
    """List of one ndarray is expanded."""
    result = _flatten_output_names([np.array(["a", "b"])])
    assert list(result) == ["a", "b"]


def test_flatten_output_names_ndarray_ndim_branch() -> None:
    """Force the ``isinstance(ndarray) and ndim > 1`` branch (line 410).

    Real ndarrays always have ``flatten``, so the first branch normally
    wins.  Temporarily making ``hasattr(..., "flatten")`` return False
    routes control to the later ndarray check while ``flatten()`` still
    works when called.
    """
    arr = np.array([["a", "b"], ["c", "d"]])
    real_hasattr = hasattr

    def _hasattr(obj: object, name: str) -> bool:
        if name == "flatten" and obj is arr:
            return False
        return real_hasattr(obj, name)

    with patch("builtins.hasattr", side_effect=_hasattr):
        result = _flatten_output_names(arr)
    assert list(result) == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------
# HTML / colour helpers
# ---------------------------------------------------------------------------


def test_build_output_tokens_html() -> None:
    """Token HTML contains uuid-scoped ids and escaped text."""
    html = _build_output_tokens_html("abc", ["hello", "<world>"])
    assert "abc_output_flat_token_0" in html
    assert "abc_output_flat_token_1" in html
    assert "&lt;world&gt;" in html


def test_grayscale_rgba() -> None:
    """RGB image is converted to 4-channel grayscale RGBA."""
    rgb = np.random.default_rng(0).integers(0, 256, size=(4, 5, 3))
    gray = _grayscale_rgba(rgb)
    assert gray.shape == (4, 5, 4)


def test_shap_color_maps() -> None:
    """Each output token maps to an RGBA grid."""
    h, w, tokens = 4, 4, 3
    values = np.random.default_rng(0).normal(size=(h, w, 1, tokens))
    exp = SimpleNamespace(values=values)
    model_output = ["t0", "t1", "t2"]
    cmap_dict = _shap_color_maps("uid", exp, model_output)
    assert len(cmap_dict) == 3
    assert "uid_output_flat_token_0" in cmap_dict


def test_image_viz_html() -> None:
    """HTML shell contains the uuid and output text."""
    html = _image_viz_html("xyz", "<span>tok</span>")
    assert "xyz_image_viz" in html
    assert "<span>tok</span>" in html


def test_image_viz_script() -> None:
    """Script contains canvas helpers and embedded JSON."""
    script = _image_viz_script(
        "xyz",
        "[[0]]",
        "[[0]]",
        "{}",
        4,
        4,
    )
    assert "xyz_redraw" in script
    assert "xyz_zoom" in script


# ---------------------------------------------------------------------------
# _shap_image_to_text
# ---------------------------------------------------------------------------


def test_shap_image_to_text_requires_ipython() -> None:
    """Missing IPython raises ImportError."""
    exp = SimpleNamespace(
        values=np.zeros((4, 4, 1, 2)),
        data=np.zeros((4, 4, 3)),
        output_names=["a", "b"],
    )
    import builtins

    real_import = builtins.__import__

    def _fake_import(
        name: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        if name.startswith("IPython"):
            raise ImportError("no ipython")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    with (
        patch("builtins.__import__", side_effect=_fake_import),
        pytest.raises(ImportError, match="IPython is required"),
    ):
        _shap_image_to_text(exp)


def test_shap_image_to_text_single() -> None:
    """Single-instance path builds HTML/script and calls display."""
    h, w, tokens = 4, 4, 2
    values = np.random.default_rng(0).normal(size=(h, w, 1, tokens))
    data = np.random.default_rng(1).integers(0, 256, size=(h, w, 3))
    exp = SimpleNamespace(
        values=values,
        data=data,
        output_names=["tok0", "tok1"],
    )
    mock_html_cls = MagicMock(side_effect=lambda content: content)
    mock_display = MagicMock()

    import builtins

    real_import = builtins.__import__

    def _import(
        name: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        if name == "IPython.display":
            mod = MagicMock()
            mod.HTML = mock_html_cls
            mod.display = mock_display
            return mod
        if name == "IPython":
            return MagicMock()
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    with patch("builtins.__import__", side_effect=_import):
        _shap_image_to_text(exp)
    mock_display.assert_called()


def test_shap_image_to_text_batch() -> None:
    """5-D values recurse over the sample axis with ordinal labels.

    ``xwhy.plots.visualisation.image`` may resolve to the ``image`` function
    via package re-exports, so the submodule is loaded with importlib.
    """
    import builtins
    import importlib

    n, h, w, tokens = 2, 4, 4, 2
    values = np.random.default_rng(0).normal(size=(n, h, w, 1, tokens))
    data = np.random.default_rng(1).integers(0, 256, size=(n, h, w, 3))

    class _BatchExp:
        def __init__(self, vals: NDArray[Any], dat: NDArray[Any]) -> None:
            self.values = vals
            self.data = dat
            self.output_names = ["t0", "t1"]

        def __getitem__(self, i: int) -> SimpleNamespace:
            return SimpleNamespace(
                values=self.values[i],
                data=self.data[i],
                output_names=self.output_names,
            )

    exp = _BatchExp(values, data)
    mock_html_cls = MagicMock(side_effect=lambda content: content)
    mock_display = MagicMock()
    child_calls: list[object] = []

    real_import = builtins.__import__
    mod = importlib.import_module("xwhy.plots.visualisation.image")
    real_fn = mod._shap_image_to_text

    def _import(
        name: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        if name == "IPython.display":
            ipy_mod = MagicMock()
            ipy_mod.HTML = mock_html_cls
            ipy_mod.display = mock_display
            return ipy_mod
        if name == "IPython":
            return MagicMock()
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    def _side_effect(sv: Any) -> None:  # noqa: ANN401
        if len(sv.values.shape) == 5:
            # Run the real batch branch; recursive calls hit this patch.
            real_fn(sv)
            return
        child_calls.append(sv)

    with (
        patch("builtins.__import__", side_effect=_import),
        patch.object(mod, "_shap_image_to_text", side_effect=_side_effect),
    ):
        mod._shap_image_to_text(exp)

    assert len(child_calls) == 2
    assert mock_display.call_count >= 2


# ---------------------------------------------------------------------------
# image (public)
# ---------------------------------------------------------------------------


def test_image_with_finish() -> None:
    """When _finish_matplotlib is available, delegate to it."""
    exp_obj = _make_explanation(1, 6, 6)
    with (
        patch(
            "xwhy.plots.visualisation.image._as_explanation",
            return_value=exp_obj,
        ),
        patch(
            "xwhy.plots.visualisation.image._finish_matplotlib",
            return_value="finished",
        ),
    ):
        result = image(
            exp_obj,
            show=True,
            title="Img",
            figsize=(10, 4),
        )
        assert result == "finished"  # type: ignore[comparison-overlap]


def test_image_show_true_no_finish() -> None:
    """show=True without _finish_matplotlib calls plt.show()."""
    exp_obj = _make_explanation(1, 6, 6)
    with (
        patch(
            "xwhy.plots.visualisation.image._as_explanation",
            return_value=exp_obj,
        ),
        patch(
            "xwhy.plots.visualisation.image._finish_matplotlib",
            new=None,
        ),
        patch("matplotlib.pyplot.show") as mock_show,
    ):
        result = image(exp_obj, show=True)
        assert result is None
        mock_show.assert_called_once()


def test_image_show_false_no_finish() -> None:
    """show=False without _finish_matplotlib returns Figure."""
    exp_obj = _make_explanation(1, 6, 6)
    with (
        patch(
            "xwhy.plots.visualisation.image._as_explanation",
            return_value=exp_obj,
        ),
        patch(
            "xwhy.plots.visualisation.image._finish_matplotlib",
            new=None,
        ),
    ):
        fig = image(exp_obj, show=False)
        assert fig is not None
        assert isinstance(fig, Figure)


def test_image_as_explanation_none() -> None:
    """_as_explanation is None; explanation used as-is."""
    exp_obj = _make_explanation(1, 6, 6)
    with (
        patch(
            "xwhy.plots.visualisation.image._as_explanation",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.image._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = image(exp_obj, show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_image_no_title_no_figsize() -> None:
    """title=None and figsize=None use defaults."""
    exp_obj = _make_explanation(1, 6, 6)
    with (
        patch(
            "xwhy.plots.visualisation.image._as_explanation",
            return_value=exp_obj,
        ),
        patch(
            "xwhy.plots.visualisation.image._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = image(exp_obj, show=False, title=None, figsize=None)
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_image_kwargs_forwarded() -> None:
    """true_labels, width, aspect, hspace, labelpad, cmap, vmax kwargs."""
    exp_obj = _make_explanation(1, 6, 6)
    cmap = LinearSegmentedColormap.from_list("rtb", ["#f00", "#00f"])
    with (
        patch(
            "xwhy.plots.visualisation.image._as_explanation",
            return_value=exp_obj,
        ),
        patch(
            "xwhy.plots.visualisation.image._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = image(
            exp_obj,
            show=False,
            true_labels=["lbl"],
            aspect=0.3,
            hspace=0.15,
            labelpad=2.0,
            cmap=cmap,
            vmax=1.0,
        )
        assert result == "ok"  # type: ignore[comparison-overlap]


def test_image_with_pixel_values_override() -> None:
    """pixel_values argument overrides explanation.data."""
    exp_obj = _make_explanation(1, 6, 6)
    other = _rgb_batch(1, 6, 6, seed=9)
    with (
        patch(
            "xwhy.plots.visualisation.image._as_explanation",
            return_value=exp_obj,
        ),
        patch(
            "xwhy.plots.visualisation.image._finish_matplotlib",
            return_value="ok",
        ),
    ):
        result = image(exp_obj, pixel_values=other, show=False)
        assert result == "ok"  # type: ignore[comparison-overlap]


# ---------------------------------------------------------------------------
# image_to_text (public)
# ---------------------------------------------------------------------------


def test_image_to_text_delegates() -> None:
    """image_to_text converts explanation and calls _shap_image_to_text."""
    exp = object()
    converted = SimpleNamespace(
        values=np.zeros((4, 4, 1, 2)),
        data=np.zeros((4, 4, 3)),
        output_names=["a", "b"],
    )
    with (
        patch(
            "xwhy.plots.visualisation.image._as_explanation",
            return_value=converted,
        ),
        patch(
            "xwhy.plots.visualisation.image._shap_image_to_text",
        ) as mock_fn,
    ):
        image_to_text(exp, max_tokens=4, show=True, title="t")
        mock_fn.assert_called_once_with(converted)


def test_image_to_text_as_explanation_none() -> None:
    """_as_explanation is None; raw explanation is forwarded."""
    exp = SimpleNamespace(
        values=np.zeros((4, 4, 1, 2)),
        data=np.zeros((4, 4, 3)),
        output_names=["a", "b"],
    )
    with (
        patch(
            "xwhy.plots.visualisation.image._as_explanation",
            new=None,
        ),
        patch(
            "xwhy.plots.visualisation.image._shap_image_to_text",
        ) as mock_fn,
    ):
        image_to_text(exp)
        mock_fn.assert_called_once_with(exp)
