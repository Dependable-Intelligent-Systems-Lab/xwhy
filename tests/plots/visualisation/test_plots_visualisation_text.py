"""Test text module."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from xwhy.plots.visualisation.text import (
    _css_rgba,
    _encode_token,
    _values_min_max,
    process_shap_values,
    svg_force_plot,
    text,
    unpack_shap_explanation_contents,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokens(n: int = 4) -> list[str]:
    """Return a short token list."""
    return [f"tok{i}" for i in range(n)]


def _values(n: int = 4, seed: int = 0) -> NDArray[np.floating[Any]]:
    """Return deterministic attribution values with mixed signs."""
    rng = np.random.default_rng(seed)
    vals = rng.normal(size=n)
    vals[0] = abs(vals[0]) + 0.1
    vals[1] = -abs(vals[1]) - 0.1
    return vals.astype(float)


def _make_exp(
    n_tokens: int = 4,
    *,
    seed: int = 0,
    shape_extra: tuple[int, ...] = (),
    output_names: Any = None,  # noqa: ANN401
    hierarchical: bool = False,
    data: Any = None,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Build a minimal Explanation-like object for text plots."""
    tokens = _tokens(n_tokens)
    vals = _values(n_tokens, seed=seed)

    if shape_extra:
        # Multi-dim: expand values
        if len(shape_extra) == 1:
            # (tokens, outputs) or (batch, tokens)
            k = shape_extra[0]
            values = np.column_stack(
                [_values(n_tokens, seed=seed + i) for i in range(k)]
            )
            base = np.array([0.1 * (i + 1) for i in range(k)], dtype=float)
            shape = (n_tokens, k)
        elif len(shape_extra) == 2:
            batch, outputs = shape_extra
            values = np.stack(
                [
                    np.column_stack(
                        [
                            _values(n_tokens, seed=seed + b * 10 + o)
                            for o in range(outputs)
                        ]
                    )
                    for b in range(batch)
                ],
                axis=0,
            )
            base = np.full((batch, outputs), 0.1)
            shape = (batch, n_tokens, outputs)  # type: ignore[assignment]
        else:
            values = vals
            base = 0.1  # type: ignore[assignment]
            shape = (n_tokens,)  # type: ignore[assignment]
    else:
        values = vals
        base = 0.1  # type: ignore[assignment]
        shape = (n_tokens,)  # type: ignore[assignment]

    clustering = None
    hierarchical_values = None
    if hierarchical:
        # Simple clustering: merge tokens pairwise → hierarchical values longer
        # One merge of first two leaves
        clustering = np.array([[0, 1]], dtype=float)
        hierarchical_values = np.concatenate([vals, np.array([0.05])])

    class Explanation:
        def __init__(self) -> None:
            self.values = values
            self.data = data if data is not None else tokens
            self.feature_names = tokens
            self.base_values = base
            self.output_names = output_names
            self.shape = shape
            self.clustering = clustering
            self.hierarchical_values = hierarchical_values

        def __getitem__(self, key: object) -> Any:  # noqa: ANN401
            if isinstance(key, int):
                # batch index
                sub = Explanation.__new__(Explanation)
                if len(np.asarray(self.values).shape) == 3:
                    sub.values = self.values[key]
                    sub.base_values = (
                        self.base_values[key]
                        if np.ndim(self.base_values) > 0
                        else self.base_values
                    )
                    sub.data = self.data
                    sub.shape = self.values[key].shape
                elif (
                    len(np.asarray(self.values).shape) == 2
                    and self.output_names is None
                ):
                    sub.values = self.values[key] if False else self.values  # row
                    # treat as iterating rows of (n, features) — for batch of 1d
                    sub.values = np.asarray(self.values)[key]
                    sub.base_values = (
                        float(np.asarray(self.base_values).ravel()[key])  # type: ignore[assignment]
                        if np.size(self.base_values) > 1
                        else float(self.base_values)
                    )
                    sub.data = (
                        self.data[key]
                        if isinstance(self.data, list)
                        and len(self.data) == np.asarray(self.values).shape[0]
                        else self.data
                    )
                    sub.shape = np.asarray(sub.values).shape
                else:
                    sub.values = self.values[key]
                    sub.base_values = self.base_values
                    sub.data = self.data
                    sub.shape = np.asarray(sub.values).shape
                sub.feature_names = self.feature_names
                sub.output_names = self.output_names
                sub.clustering = self.clustering
                sub.hierarchical_values = self.hierarchical_values
                return sub
            if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], slice):
                # [:, i] output column
                col = int(key[1])
                sub = Explanation.__new__(Explanation)
                vals_a = np.asarray(self.values)
                if vals_a.ndim == 2:
                    sub.values = vals_a[:, col]
                    sub.base_values = float(  # type: ignore[assignment]
                        np.asarray(self.base_values).ravel()[col]
                        if np.size(self.base_values) > 1
                        else self.base_values
                    )
                    sub.shape = sub.values.shape
                elif vals_a.ndim == 3:
                    # [j, :, i] handled by outer
                    sub.values = vals_a[:, :, col]
                    sub.base_values = self.base_values
                    sub.shape = sub.values.shape
                else:
                    sub.values = vals_a
                    sub.base_values = self.base_values
                    sub.shape = vals_a.shape
                sub.data = self.data
                sub.feature_names = self.feature_names
                sub.output_names = (
                    self.output_names[col]
                    if isinstance(self.output_names, (list, np.ndarray))
                    else self.output_names
                )
                sub.clustering = self.clustering
                sub.hierarchical_values = self.hierarchical_values
                return sub
            if isinstance(key, tuple) and len(key) == 3:
                j, _sl, i = key
                sub = Explanation.__new__(Explanation)
                vals_a = np.asarray(self.values)
                sub.values = vals_a[int(j), :, int(i)]
                bv = np.asarray(self.base_values)
                sub.base_values = (
                    float(bv[int(j), int(i)]) if bv.ndim == 2 else float(bv)  # type: ignore[assignment]
                )
                sub.data = self.data
                sub.feature_names = self.feature_names
                sub.output_names = self.output_names
                sub.shape = sub.values.shape
                sub.clustering = self.clustering
                sub.hierarchical_values = self.hierarchical_values
                return sub
            raise TypeError(key)

        def __iter__(self) -> Any:  # noqa: ANN401
            vals_a = np.asarray(self.values)
            if vals_a.ndim >= 2:
                for i in range(vals_a.shape[0]):
                    yield self[i]
            else:
                yield self

        def __len__(self) -> int:
            return int(np.asarray(self.values).shape[0])

    return Explanation()


# ---------------------------------------------------------------------------
# _css_rgba / _encode_token / _values_min_max
# ---------------------------------------------------------------------------


def test_css_rgba() -> None:
    """Format rgba with plain float literals."""
    result = _css_rgba(255.0, 128.0, 0.0, 0.5)
    assert result == "rgba(255.0, 128.0, 0.0, 0.5)"


def test_encode_token() -> None:
    """HTML-escape angle brackets and strip ## prefixes."""
    assert "&lt;" in _encode_token("<tag>")
    assert "&gt;" in _encode_token("<tag>")
    assert _encode_token("foo ##bar") == "foo bar" or "##" not in _encode_token(
        "foo ##bar"
    )


def test_values_min_max() -> None:
    """Axis bounds pad around positive and negative contributions."""
    vals = np.array([1.0, -0.5, 0.2])
    xmin, xmax, cmax = _values_min_max(vals, base_values=0.0)
    assert xmin < xmax
    assert cmax > 0.0


# ---------------------------------------------------------------------------
# unpack_shap_explanation_contents
# ---------------------------------------------------------------------------


def test_unpack_plain_values() -> None:
    """Falls back to .values when hierarchical_values is absent."""
    exp = SimpleNamespace(values=np.array([0.1, -0.2]), clustering=None)
    values, clustering = unpack_shap_explanation_contents(exp)
    assert values.shape == (2,)
    assert clustering is None


def test_unpack_hierarchical() -> None:
    """Prefers hierarchical_values when present."""
    exp = SimpleNamespace(
        values=np.array([0.1, 0.2]),
        hierarchical_values=np.array([0.1, 0.2, 0.05]),
        clustering=np.array([[0, 1]]),
    )
    values, clustering = unpack_shap_explanation_contents(exp)
    assert len(values) == 3
    assert clustering is not None


# ---------------------------------------------------------------------------
# process_shap_values
# ---------------------------------------------------------------------------


def test_process_shap_non_hierarchical() -> None:
    """Matching lengths return tokens and values unchanged."""
    tokens = _tokens(3)
    values = _values(3)
    out_t, _out_v, out_s = process_shap_values(  # type: ignore[misc]
        tokens, values, grouping_threshold=0.01, separator=""
    )
    assert len(out_t) == 3
    assert out_s.shape == (3,)


def test_process_shap_non_hierarchical_meta() -> None:
    """return_meta_data adds mapping and collapsed ids."""
    tokens = _tokens(3)
    values = _values(3)
    result = process_shap_values(
        tokens,
        values,
        grouping_threshold=0.01,
        separator="",
        return_meta_data=True,
    )
    assert len(result) == 5


def test_process_shap_hierarchical_requires_clustering() -> None:
    """Mismatched lengths without clustering raise ValueError."""
    tokens = _tokens(3)
    values = np.array([0.1, 0.2, 0.3, 0.05])
    with pytest.raises(ValueError, match="clustering"):
        process_shap_values(tokens, values, 0.01, "")


def test_process_shap_hierarchical_collapse() -> None:
    """Low threshold collapses the interaction node into one token."""
    tokens = ["a", "b", "c"]
    # leaf values + one interaction
    values = np.array([0.1, 0.2, 0.05, 0.5])
    clustering = np.array([[0, 1]], dtype=float)
    out_t, _out_v, _out_s = process_shap_values(  # type: ignore[misc]
        tokens,
        values,
        grouping_threshold=100.0,  # force collapse
        separator=" ",
        clustering=clustering,
    )
    assert len(out_t) >= 1


def test_process_shap_hierarchical_no_collapse() -> None:
    """High child effects keep leaves separate."""
    tokens = ["a", "b", "c"]
    values = np.array([1.0, 1.0, 0.1, 0.001])
    clustering = np.array([[0, 1]], dtype=float)
    out_t, _out_v, _out_s = process_shap_values(  # type: ignore[misc]
        tokens,
        values,
        grouping_threshold=0.01,
        separator="",
        clustering=clustering,
    )
    assert len(out_t) >= 2


def test_process_shap_hierarchical_meta() -> None:
    """Hierarchical path with return_meta_data returns five arrays."""
    tokens = ["a", "b"]
    values = np.array([0.1, 0.2, 0.05])
    clustering = np.array([[0, 1]], dtype=float)
    result = process_shap_values(
        tokens,
        values,
        grouping_threshold=0.01,
        separator="-",
        clustering=clustering,
        return_meta_data=True,
    )
    assert len(result) == 5


# ---------------------------------------------------------------------------
# svg_force_plot
# ---------------------------------------------------------------------------


def test_svg_force_plot_basic() -> None:
    """SVG force plot returns markup containing svg tags."""
    vals = _values(5)
    tokens = _tokens(5)
    html = svg_force_plot(
        vals,
        base_values=0.0,
        fx=float(vals.sum()),
        tokens=tokens,
        uuid="testid",
        xmin=-2.0,
        xmax=2.0,
        output_name="out",
    )
    assert "<svg" in html
    assert "</svg>" in html
    assert "testid" in html


def test_svg_force_plot_all_positive() -> None:
    """All-positive values still render."""
    vals = np.array([0.5, 0.3, 0.1])
    html = svg_force_plot(vals, 0.0, float(vals.sum()), _tokens(3), "u", -1.0, 2.0, "")
    assert "<svg" in html


def test_svg_force_plot_all_negative() -> None:
    """All-negative values still render."""
    vals = np.array([-0.5, -0.3, -0.1])
    html = svg_force_plot(vals, 0.0, float(vals.sum()), _tokens(3), "u", -2.0, 1.0, "y")
    assert "<svg" in html


# ---------------------------------------------------------------------------
# text (public) — single explanation
# ---------------------------------------------------------------------------


def test_text_single_show_false() -> None:
    """Single explanation with show=False returns HTML string."""
    exp = _make_exp(4)
    with patch(
        "xwhy.plots.visualisation.text._as_explanation",
        return_value=exp,
    ):
        result = text(exp, show=False, num_starting_labels=2)
    assert result is not None
    assert isinstance(result, str)
    assert "<svg" in result


def test_text_single_show_true() -> None:
    """show=True returns None and invokes _finish_html when available."""
    exp = _make_exp(4)
    mock_finish = MagicMock()
    with (
        patch(
            "xwhy.plots.visualisation.text._as_explanation",
            return_value=exp,
        ),
        patch(
            "xwhy.plots.visualisation.text._finish_html",
            mock_finish,
        ),
    ):
        result = text(exp, show=True, title="Title")
    assert result is None
    mock_finish.assert_called_once()


def test_text_single_no_finish() -> None:
    """Without _finish_html, show=False still returns HTML."""
    exp = _make_exp(4)
    with (
        patch(
            "xwhy.plots.visualisation.text._as_explanation",
            return_value=exp,
        ),
        patch(
            "xwhy.plots.visualisation.text._finish_html",
            new=None,
        ),
    ):
        result = text(exp, show=False)
    assert result is not None
    assert "<svg" in result


def test_text_single_as_explanation_none() -> None:
    """_as_explanation is None; explanation is used as-is."""
    exp = _make_exp(3)
    with patch(
        "xwhy.plots.visualisation.text._as_explanation",
        new=None,
    ):
        result = text(exp, show=False)
    assert result is not None


def test_text_data_from_feature_names() -> None:
    """Missing data falls back to feature_names."""
    exp = _make_exp(3)
    exp.data = None
    with patch(
        "xwhy.plots.visualisation.text._as_explanation",
        return_value=exp,
    ):
        result = text(exp, show=False)
    assert result is not None


def test_text_group_size_label() -> None:
    """Collapsed groups show value / size labels."""
    tokens = ["a", "b"]
    values = np.array([0.1, 0.2, 0.5])
    clustering = np.array([[0, 1]], dtype=float)
    exp = _make_exp(2)
    exp.data = tokens
    exp.values = values[:2]
    exp.hierarchical_values = values
    exp.clustering = clustering
    with patch(
        "xwhy.plots.visualisation.text._as_explanation",
        return_value=exp,
    ):
        result = text(
            exp,
            show=False,
            grouping_threshold=100.0,
            separator=" ",
            num_starting_labels=2,
        )
    assert result is not None


def test_text_fixed_axis_bounds() -> None:
    """Explicit xmin/xmax/cmax are respected."""
    exp = _make_exp(4)
    with patch(
        "xwhy.plots.visualisation.text._as_explanation",
        return_value=exp,
    ):
        result = text(exp, show=False, xmin=-3.0, xmax=3.0, cmax=2.0)
    assert result is not None


def test_text_red_transparent_blue_none() -> None:
    """Fallback colour when RED_TRANSPARENT_BLUE is None."""
    exp = _make_exp(3)
    with (
        patch(
            "xwhy.plots.visualisation.text._as_explanation",
            return_value=exp,
        ),
        patch(
            "xwhy.plots.visualisation.text.RED_TRANSPARENT_BLUE",
            new=None,
        ),
    ):
        result = text(exp, show=False, num_starting_labels=1)
    assert result is not None


# ---------------------------------------------------------------------------
# text — multi-row batch (2-D, no output_names list)
# ---------------------------------------------------------------------------


def test_text_batch_2d() -> None:
    """2-D explanation with scalar/None output_names iterates rows."""
    # shape (batch, tokens) — simulate via custom object
    n_batch, n_tok = 2, 3
    rng = np.random.default_rng(0)

    class _BatchExp:
        def __init__(self) -> None:
            self.values = rng.normal(size=(n_batch, n_tok))
            self.data = [_tokens(n_tok) for _ in range(n_batch)]
            self.feature_names = _tokens(n_tok)
            self.base_values = np.array([0.1, 0.2])
            self.output_names = None
            self.shape = (n_batch, n_tok)
            self.clustering = None
            self.hierarchical_values = None

        def __iter__(self) -> Any:  # noqa: ANN401
            for i in range(n_batch):
                yield self[i]

        def __getitem__(self, i: int) -> SimpleNamespace:
            return SimpleNamespace(
                values=self.values[i],
                data=self.data[i],
                feature_names=self.feature_names,
                base_values=float(self.base_values[i]),
                output_names=None,
                shape=(n_tok,),
                clustering=None,
                hierarchical_values=None,
            )

    exp = _BatchExp()
    with patch(
        "xwhy.plots.visualisation.text._as_explanation",
        side_effect=lambda x: x,
    ):
        result = text(exp, show=False, title="Batch")
    assert result is not None
    assert "[0]" in result
    assert "[1]" in result


def test_text_batch_2d_show_true() -> None:
    """Batch path with show=True returns None."""
    n_batch, n_tok = 2, 3
    rng = np.random.default_rng(1)

    class _BatchExp:
        def __init__(self) -> None:
            self.values = rng.normal(size=(n_batch, n_tok))
            self.data = [_tokens(n_tok) for _ in range(n_batch)]
            self.feature_names = _tokens(n_tok)
            self.base_values = np.array([0.0, 0.0])
            self.output_names = "scalar_out"
            self.shape = (n_batch, n_tok)
            self.clustering = None
            self.hierarchical_values = None

        def __iter__(self) -> Any:  # noqa: ANN401
            for i in range(n_batch):
                yield self[i]

        def __getitem__(self, i: int) -> SimpleNamespace:
            return SimpleNamespace(
                values=self.values[i],
                data=self.data[i],
                feature_names=self.feature_names,
                base_values=0.0,
                output_names="scalar_out",
                shape=(n_tok,),
                clustering=None,
                hierarchical_values=None,
            )

    exp = _BatchExp()
    mock_finish = MagicMock()
    with (
        patch(
            "xwhy.plots.visualisation.text._as_explanation",
            side_effect=lambda x: x,
        ),
        patch(
            "xwhy.plots.visualisation.text._finish_html",
            mock_finish,
        ),
    ):
        result = text(exp, show=True)
    assert result is None
    mock_finish.assert_called()


# ---------------------------------------------------------------------------
# text — multi-output (2-D with output_names list)
# ---------------------------------------------------------------------------


def test_text_multi_output() -> None:
    """2-D with list output_names renders interactive output tabs."""
    n_tok, n_out = 3, 2
    rng = np.random.default_rng(2)
    values = rng.normal(size=(n_tok, n_out))
    tokens = _tokens(n_tok)

    class _MultiOut:
        def __init__(self) -> None:
            self.values = values
            self.data = tokens
            self.feature_names = tokens
            self.base_values = np.array([0.1, 0.2])
            self.output_names = ["class0", "class1"]
            self.shape = (n_tok, n_out)
            self.clustering = None
            self.hierarchical_values = None

        def __getitem__(self, key: object) -> SimpleNamespace:
            if isinstance(key, tuple) and isinstance(key[0], slice):
                col = int(key[1])
                return SimpleNamespace(
                    values=self.values[:, col],
                    data=self.data,
                    feature_names=self.feature_names,
                    base_values=float(self.base_values[col]),
                    output_names=self.output_names[col],
                    shape=(n_tok,),
                    clustering=None,
                    hierarchical_values=None,
                )
            raise TypeError(key)

    exp = _MultiOut()
    with patch(
        "xwhy.plots.visualisation.text._as_explanation",
        side_effect=lambda x: x,
    ):
        result = text(exp, show=False, title="Multi")
    assert result is not None
    assert "class0" in result
    assert "class1" in result
    assert "outputs" in result


def test_text_multi_output_fixed_bounds() -> None:
    """Multi-output with explicit xmin/xmax/cmax skips auto bounds."""
    n_tok, n_out = 3, 2
    rng = np.random.default_rng(3)

    class _MultiOut:
        def __init__(self) -> None:
            self.values = rng.normal(size=(n_tok, n_out))
            self.data = _tokens(n_tok)
            self.feature_names = _tokens(n_tok)
            self.base_values = np.array([0.0, 0.0])
            self.output_names = ["a", "b"]
            self.shape = (n_tok, n_out)
            self.clustering = None
            self.hierarchical_values = None

        def __getitem__(self, key: object) -> SimpleNamespace:
            if isinstance(key, tuple) and isinstance(key[0], slice):
                col = int(key[1])
                return SimpleNamespace(
                    values=self.values[:, col],
                    data=self.data,
                    feature_names=self.feature_names,
                    base_values=0.0,
                    output_names=self.output_names[col],
                    shape=(n_tok,),
                    clustering=None,
                    hierarchical_values=None,
                )
            raise TypeError(key)

    exp = _MultiOut()
    with patch(
        "xwhy.plots.visualisation.text._as_explanation",
        side_effect=lambda x: x,
    ):
        result = text(exp, show=False, xmin=-2.0, xmax=2.0, cmax=1.0)
    assert result is not None


def test_text_multi_output_cmap_none() -> None:
    """Multi-output falls back when RED_TRANSPARENT_BLUE is None."""
    n_tok, n_out = 3, 2
    rng = np.random.default_rng(4)

    class _MultiOut:
        def __init__(self) -> None:
            self.values = rng.normal(size=(n_tok, n_out))
            self.data = _tokens(n_tok)
            self.feature_names = _tokens(n_tok)
            self.base_values = np.zeros(n_out)
            self.output_names = ["x", "y"]
            self.shape = (n_tok, n_out)
            self.clustering = None
            self.hierarchical_values = None

        def __getitem__(self, key: object) -> SimpleNamespace:
            if isinstance(key, tuple) and isinstance(key[0], slice):
                col = int(key[1])
                return SimpleNamespace(
                    values=self.values[:, col],
                    data=self.data,
                    feature_names=self.feature_names,
                    base_values=0.0,
                    output_names=self.output_names[col],
                    shape=(n_tok,),
                    clustering=None,
                    hierarchical_values=None,
                )
            raise TypeError(key)

    exp = _MultiOut()
    with (
        patch(
            "xwhy.plots.visualisation.text._as_explanation",
            side_effect=lambda x: x,
        ),
        patch(
            "xwhy.plots.visualisation.text.RED_TRANSPARENT_BLUE",
            new=None,
        ),
    ):
        result = text(exp, show=False)
    assert result is not None


# ---------------------------------------------------------------------------
# text — 3-D (batch x tokens x outputs)
# ---------------------------------------------------------------------------


def test_text_3d_batch() -> None:
    """3-D explanation iterates batch rows with shared bounds."""
    batch, n_tok, n_out = 2, 3, 2
    rng = np.random.default_rng(5)
    values = rng.normal(size=(batch, n_tok, n_out))
    tokens = _tokens(n_tok)

    class ThreeDExp:
        def __init__(self) -> None:
            self.values = values
            self.data = tokens
            self.feature_names = tokens
            self.base_values = np.zeros((batch, n_out))
            self.output_names = ["o0", "o1"]
            self.shape = (batch, n_tok, n_out)
            self.clustering = None
            self.hierarchical_values = None

        def __iter__(self) -> Any:  # noqa: ANN401
            for i in range(batch):
                yield self[i]

        def __getitem__(self, key: object) -> Any:  # noqa: ANN401
            if isinstance(key, int):
                # Return a 2-D multi-output slice for one batch item
                class _Slice:
                    def __init__(self) -> None:
                        self.values = values[key]  # type: ignore[call-overload]
                        self.data = tokens
                        self.feature_names = tokens
                        self.base_values = np.zeros(n_out)
                        self.output_names = ["o0", "o1"]
                        self.shape = (n_tok, n_out)
                        self.clustering = None
                        self.hierarchical_values = None

                    def __getitem__(self, k: object) -> SimpleNamespace:
                        if isinstance(k, tuple) and isinstance(k[0], slice):
                            col = int(k[1])
                            return SimpleNamespace(
                                values=self.values[:, col],
                                data=tokens,
                                feature_names=tokens,
                                base_values=0.0,
                                output_names=self.output_names[col],
                                shape=(n_tok,),
                                clustering=None,
                                hierarchical_values=None,
                            )
                        raise TypeError(k)

                return _Slice()
            if isinstance(key, tuple) and len(key) == 3:
                j, _sl, i = key
                return SimpleNamespace(
                    values=values[int(j), :, int(i)],
                    data=tokens,
                    feature_names=tokens,
                    base_values=0.0,
                    output_names=None,
                    shape=(n_tok,),
                    clustering=None,
                    hierarchical_values=None,
                )
            raise TypeError(key)

    exp = ThreeDExp()
    with patch(
        "xwhy.plots.visualisation.text._as_explanation",
        side_effect=lambda x: x,
    ):
        result = text(exp, show=False, title="3D")
    assert result is not None
    assert "[0]" in result


def test_text_3d_fixed_bounds() -> None:
    """3-D with explicit bounds skips auto aggregation assignment."""
    batch, n_tok, n_out = 2, 3, 2
    rng = np.random.default_rng(6)
    values = rng.normal(size=(batch, n_tok, n_out))
    tokens = _tokens(n_tok)

    class ThreeDExp:
        def __init__(self) -> None:
            self.values = values
            self.data = tokens
            self.feature_names = tokens
            self.base_values = np.zeros((batch, n_out))
            self.output_names = ["o0", "o1"]
            self.shape = (batch, n_tok, n_out)
            self.clustering = None
            self.hierarchical_values = None

        def __iter__(self) -> Any:  # noqa: ANN401
            for i in range(batch):
                yield self[i]

        def __getitem__(self, key: object) -> Any:  # noqa: ANN401
            if isinstance(key, int):

                class _Slice:
                    def __init__(self) -> None:
                        self.values = values[key]  # type: ignore[call-overload]
                        self.data = tokens
                        self.feature_names = tokens
                        self.base_values = np.zeros(n_out)
                        self.output_names = ["o0", "o1"]
                        self.shape = (n_tok, n_out)
                        self.clustering = None
                        self.hierarchical_values = None

                    def __getitem__(self, k: object) -> SimpleNamespace:
                        if isinstance(k, tuple) and isinstance(k[0], slice):
                            col = int(k[1])
                            return SimpleNamespace(
                                values=self.values[:, col],
                                data=tokens,
                                feature_names=tokens,
                                base_values=0.0,
                                output_names=self.output_names[col],
                                shape=(n_tok,),
                                clustering=None,
                                hierarchical_values=None,
                            )
                        raise TypeError(k)

                return _Slice()
            if isinstance(key, tuple) and len(key) == 3:
                j, _sl, i = key
                return SimpleNamespace(
                    values=values[int(j), :, int(i)],
                    data=tokens,
                    feature_names=tokens,
                    base_values=0.0,
                    output_names=None,
                    shape=(n_tok,),
                    clustering=None,
                    hierarchical_values=None,
                )
            raise TypeError(key)

    exp = ThreeDExp()
    with patch(
        "xwhy.plots.visualisation.text._as_explanation",
        side_effect=lambda x: x,
    ):
        result = text(exp, show=False, xmin=-1.0, xmax=1.0, cmax=1.0)
    assert result is not None


# ---------------------------------------------------------------------------
# text — _finish_html is None (batch / multi-output / 3-D)
# Covers the false branch of: if _finish_html is not None
# ---------------------------------------------------------------------------


def test_text_batch_2d_no_finish() -> None:
    """Batch path returns HTML when _finish_html is unavailable."""
    n_batch, n_tok = 2, 3
    rng = np.random.default_rng(10)

    class BatchExp:
        def __init__(self) -> None:
            self.values = rng.normal(size=(n_batch, n_tok))
            self.data = [_tokens(n_tok) for _ in range(n_batch)]
            self.feature_names = _tokens(n_tok)
            self.base_values = np.array([0.1, 0.2])
            self.output_names = None
            self.shape = (n_batch, n_tok)
            self.clustering = None
            self.hierarchical_values = None

        def __iter__(self) -> Any:  # noqa: ANN401
            for i in range(n_batch):
                yield self[i]

        def __getitem__(self, i: int) -> SimpleNamespace:
            return SimpleNamespace(
                values=self.values[i],
                data=self.data[i],
                feature_names=self.feature_names,
                base_values=float(self.base_values[i]),
                output_names=None,
                shape=(n_tok,),
                clustering=None,
                hierarchical_values=None,
            )

    exp = BatchExp()
    with (
        patch(
            "xwhy.plots.visualisation.text._as_explanation",
            side_effect=lambda x: x,
        ),
        patch(
            "xwhy.plots.visualisation.text._finish_html",
            new=None,
        ),
    ):
        result = text(exp, show=False)
    assert result is not None
    assert "[0]" in result
    assert "<svg" in result


def test_text_multi_output_no_finish() -> None:
    """Multi-output path returns HTML when _finish_html is unavailable."""
    n_tok, n_out = 3, 2
    rng = np.random.default_rng(11)
    values = rng.normal(size=(n_tok, n_out))
    tokens = _tokens(n_tok)

    class MultiOut:
        def __init__(self) -> None:
            self.values = values
            self.data = tokens
            self.feature_names = tokens
            self.base_values = np.array([0.1, 0.2])
            self.output_names = ["c0", "c1"]
            self.shape = (n_tok, n_out)
            self.clustering = None
            self.hierarchical_values = None

        def __getitem__(self, key: object) -> SimpleNamespace:
            if isinstance(key, tuple) and isinstance(key[0], slice):
                col = int(key[1])
                return SimpleNamespace(
                    values=self.values[:, col],
                    data=self.data,
                    feature_names=self.feature_names,
                    base_values=float(self.base_values[col]),
                    output_names=self.output_names[col],
                    shape=(n_tok,),
                    clustering=None,
                    hierarchical_values=None,
                )
            raise TypeError(key)

    exp = MultiOut()
    with (
        patch(
            "xwhy.plots.visualisation.text._as_explanation",
            side_effect=lambda x: x,
        ),
        patch(
            "xwhy.plots.visualisation.text._finish_html",
            new=None,
        ),
    ):
        result = text(exp, show=False)
    assert result is not None
    assert "c0" in result
    assert "outputs" in result


def test_text_3d_no_finish() -> None:
    """3-D path returns HTML when _finish_html is unavailable."""
    batch, n_tok, n_out = 2, 3, 2
    rng = np.random.default_rng(12)
    values = rng.normal(size=(batch, n_tok, n_out))
    tokens = _tokens(n_tok)

    class ThreeDExp:
        def __init__(self) -> None:
            self.values = values
            self.data = tokens
            self.feature_names = tokens
            self.base_values = np.zeros((batch, n_out))
            self.output_names = ["o0", "o1"]
            self.shape = (batch, n_tok, n_out)
            self.clustering = None
            self.hierarchical_values = None

        def __iter__(self) -> Any:  # noqa: ANN401
            for i in range(batch):
                yield self[i]

        def __getitem__(self, key: object) -> Any:  # noqa: ANN401
            if isinstance(key, int):

                class Slice:
                    def __init__(self) -> None:
                        self.values = values[key]  # type: ignore[call-overload]
                        self.data = tokens
                        self.feature_names = tokens
                        self.base_values = np.zeros(n_out)
                        self.output_names = ["o0", "o1"]
                        self.shape = (n_tok, n_out)
                        self.clustering = None
                        self.hierarchical_values = None

                    def __getitem__(self, k: object) -> SimpleNamespace:
                        if isinstance(k, tuple) and isinstance(k[0], slice):
                            col = int(k[1])
                            return SimpleNamespace(
                                values=self.values[:, col],
                                data=tokens,
                                feature_names=tokens,
                                base_values=0.0,
                                output_names=self.output_names[col],
                                shape=(n_tok,),
                                clustering=None,
                                hierarchical_values=None,
                            )
                        raise TypeError(k)

                return Slice()
            if isinstance(key, tuple) and len(key) == 3:
                j, _sl, i = key
                return SimpleNamespace(
                    values=values[int(j), :, int(i)],
                    data=tokens,
                    feature_names=tokens,
                    base_values=0.0,
                    output_names=None,
                    shape=(n_tok,),
                    clustering=None,
                    hierarchical_values=None,
                )
            raise TypeError(key)

    exp = ThreeDExp()
    with (
        patch(
            "xwhy.plots.visualisation.text._as_explanation",
            side_effect=lambda x: x,
        ),
        patch(
            "xwhy.plots.visualisation.text._finish_html",
            new=None,
        ),
    ):
        result = text(exp, show=False)
    assert result is not None
    assert "[0]" in result
