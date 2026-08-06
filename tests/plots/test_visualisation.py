"""Unit tests for the native visualisation engine."""

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pytest
from matplotlib.patches import Rectangle

from xwhy.plots import visualisation as viz
from xwhy.plots.visualisation import Explanation

matplotlib.use("Agg")

N_INSTANCES = 60
N_FEATURES = 6


@pytest.fixture(autouse=True)
def clean_plots() -> Generator[None, None, None]:
    """Ensure all matplotlib figures are closed before and after each test."""
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def rng() -> np.random.Generator:
    """Provide a deterministic random generator."""
    return np.random.default_rng(7)


@pytest.fixture
def feature_names() -> list[str]:
    """Provide feature names for the synthetic explanations."""
    return [f"feat_{i}" for i in range(N_FEATURES)]


@pytest.fixture
def data(rng: np.random.Generator) -> np.ndarray:
    """Provide a synthetic feature matrix."""
    return rng.normal(size=(N_INSTANCES, N_FEATURES))


@pytest.fixture
def values(rng: np.random.Generator, data: np.ndarray) -> np.ndarray:
    """Provide synthetic attributions correlated with the feature matrix."""
    return data * rng.normal(size=N_FEATURES) + rng.normal(
        scale=0.1, size=(N_INSTANCES, N_FEATURES)
    )


@pytest.fixture
def exp_2d(
    values: np.ndarray, data: np.ndarray, feature_names: list[str]
) -> Explanation:
    """Provide a batched explanation."""
    return Explanation(
        values=values, base_values=0.4, data=data, feature_names=feature_names
    )


@pytest.fixture
def exp_1d(
    values: np.ndarray, data: np.ndarray, feature_names: list[str]
) -> Explanation:
    """Provide a single-instance explanation."""
    return Explanation(
        values=values[0], base_values=0.4, data=data[0], feature_names=feature_names
    )


@pytest.fixture
def exp_image(rng: np.random.Generator) -> Explanation:
    """Provide an image explanation."""
    return Explanation(
        values=rng.normal(size=(2, 12, 12, 3)), data=rng.random((2, 12, 12, 3))
    )


@pytest.fixture
def exp_multimodal(rng: np.random.Generator) -> Explanation:
    """Provide a multimodal image-to-text explanation."""
    return Explanation(
        values=rng.normal(size=(1, 12, 12, 3, 4)),
        data=rng.random((1, 12, 12, 3)),
        output_names=["a", "cat", "sat", "."],
    )


# ==============================================================================
# EXPLANATION CONTAINER
# ==============================================================================


def test_explanation_shape_and_len(exp_2d: Explanation) -> None:
    """Verify shape, ndim and length mirror the underlying array."""
    assert exp_2d.shape == (N_INSTANCES, N_FEATURES)
    assert exp_2d.ndim == 2
    assert len(exp_2d) == N_INSTANCES


def test_explanation_len_of_scalar() -> None:
    """Verify a zero-dimensional explanation reports zero length."""
    assert len(Explanation(values=np.array(1.0))) == 0


def test_explanation_getitem_row(exp_2d: Explanation) -> None:
    """Verify integer indexing selects one instance and keeps the names."""
    row = exp_2d[0]
    assert row.values.shape == (N_FEATURES,)
    assert row.feature_names is not None


def test_explanation_getitem_column(exp_2d: Explanation) -> None:
    """Verify tuple indexing slices both the values and the feature names."""
    column = exp_2d[:, 2]
    assert column.values.shape == (N_INSTANCES,)
    assert str(column.feature_names) == "feat_2"


def test_explanation_getitem_with_array_base_values(values: np.ndarray) -> None:
    """Verify per-instance base values are sliced alongside the attributions."""
    exp = Explanation(values=values, base_values=np.arange(N_INSTANCES, dtype=float))
    assert exp[3].base_values == pytest.approx(3.0)


def test_explanation_getitem_with_unsliceable_data(values: np.ndarray) -> None:
    """Verify data that cannot be sliced degrades to None instead of raising."""
    exp = Explanation(values=values, data=np.array([1.0, 2.0]))
    assert exp[:, 0].data is None


def test_explanation_abs(exp_2d: Explanation, values: np.ndarray) -> None:
    """Verify the abs property returns absolute attributions."""
    np.testing.assert_allclose(exp_2d.abs.values, np.abs(values))


@pytest.mark.parametrize(
    ("method", "expected"),
    [("mean", np.mean), ("sum", np.sum), ("max", np.max), ("min", np.min)],
)
def test_explanation_reductions(
    exp_2d: Explanation,
    values: np.ndarray,
    method: str,
    expected: Callable[..., Any],
) -> None:
    """Verify each reduction collapses the instance axis correctly."""
    reduced = getattr(exp_2d, method)(0)
    np.testing.assert_allclose(reduced.values, expected(values, axis=0))
    assert reduced.feature_names is not None


def test_explanation_reduction_drops_names_when_collapsing_features(
    exp_2d: Explanation,
) -> None:
    """Verify reducing over every axis discards the now-meaningless names."""
    assert exp_2d.mean().feature_names is None


def test_explanation_repr(exp_2d: Explanation) -> None:
    """Verify the representation reports the array shape."""
    assert f"values={(N_INSTANCES, N_FEATURES)}" in repr(exp_2d)


# ==============================================================================
# CONVERSION AND HELPERS
# ==============================================================================


def test_as_explanation_passthrough(exp_2d: Explanation) -> None:
    """Verify an Explanation is returned unchanged."""
    assert viz._as_explanation(exp_2d) is exp_2d


def test_as_explanation_from_array(values: np.ndarray) -> None:
    """Verify a bare numpy array is wrapped."""
    np.testing.assert_allclose(viz._as_explanation(values).values, values)


def test_as_explanation_from_result_object(exp_2d: Explanation) -> None:
    """Verify an object exposing to_explanation is converted."""
    result = MagicMock()
    result.to_explanation.return_value = exp_2d
    assert viz._as_explanation(result) is exp_2d


def test_as_explanation_from_legacy_to_shap(exp_2d: Explanation) -> None:
    """Verify the deprecated to_shap hook is still honoured."""
    result = MagicMock(spec=["to_shap"])
    result.to_shap.return_value = exp_2d
    assert viz._as_explanation(result) is exp_2d


def test_as_explanation_from_foreign_explanation(values: np.ndarray) -> None:
    """Verify a third-party Explanation-like object is copied field by field."""
    foreign = MagicMock(spec=["values", "base_values", "data", "feature_names"])
    foreign.values = values
    foreign.base_values = 0.5
    foreign.data = None
    foreign.feature_names = None

    result = MagicMock(spec=["to_explanation"])
    result.to_explanation.return_value = foreign

    converted = viz._as_explanation(result)
    assert isinstance(converted, Explanation)
    assert converted.base_values == 0.5


def test_as_explanation_rejects_unsupported_type() -> None:
    """Verify an unusable object raises TypeError."""
    with pytest.raises(TypeError, match="Cannot build an Explanation"):
        viz._as_explanation("not an explanation")


def test_resolve_names_generates_missing() -> None:
    """Verify absent names are generated and short lists are padded."""
    assert viz._resolve_names(None, 2) == ["Feature 0", "Feature 1"]
    assert viz._resolve_names(["a"], 3) == ["a", "Feature 1", "Feature 2"]
    assert viz._resolve_names(["a", "b", "c"], 2) == ["a", "b"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1.5, "1.5"), (2.0, "2"), (None, ""), ("text", "text"), (-0.001, "0")],
)
def test_format_value(value: Any, expected: str) -> None:  # noqa: ANN401
    """Verify display formatting trims redundant zeros and handles non-numerics."""
    assert viz._format_value(value) == expected


def test_format_value_of_unformattable_object() -> None:
    """Verify an object that cannot be coerced falls back to str()."""
    assert viz._format_value(object) == str(object)


def test_group_minor_features_preserves_total(feature_names: list[str]) -> None:
    """Verify grouping the tail keeps the attributions summing to the same total."""
    scores = np.array([5.0, -4.0, 3.0, -2.0, 1.0, 0.5])
    grouped, labels = viz._group_minor_features(scores, feature_names, 3)

    assert len(grouped) == 3
    assert grouped.sum() == pytest.approx(scores.sum())
    assert "Sum of 4 other features" in labels


def test_group_minor_features_without_grouping(feature_names: list[str]) -> None:
    """Verify max_display=None keeps every feature."""
    scores = np.arange(N_FEATURES, dtype=float)
    grouped, labels = viz._group_minor_features(scores, feature_names, None)

    assert len(grouped) == N_FEATURES
    assert len(labels) == N_FEATURES


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        ("matplotlib", "matplotlib"),
        ("MPL", "matplotlib"),
        ("plotly", "plotly"),
        ("go", "plotly"),
        ("html", "html"),
    ],
)
def test_check_backend_canonicalises(backend: str, expected: str) -> None:
    """Verify backend aliases resolve to their canonical name."""
    allowed = frozenset({"matplotlib", "plotly", "html"})
    assert viz._check_backend(backend, allowed) == expected


def test_check_backend_rejects_unknown() -> None:
    """Verify an unknown backend raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        viz._check_backend("ggplot", frozenset({"matplotlib"}))


def test_check_backend_rejects_unsupported() -> None:
    """Verify a known but unsupported backend raises ValueError."""
    with pytest.raises(ValueError, match="not supported by this plot"):
        viz._check_backend("plotly", frozenset({"matplotlib"}))


def test_robust_limits_handles_degenerate_input() -> None:
    """Verify constant, empty and None inputs still yield a usable range."""
    assert viz._robust_limits(None) == (0.0, 1.0)
    assert viz._robust_limits(np.array([])) == (0.0, 1.0)

    low, high = viz._robust_limits(np.full(10, 3.0))
    assert low < high


def test_density_offsets_shape_and_range(values: np.ndarray) -> None:
    """Verify beeswarm offsets stay inside the row and match the input length."""
    offsets = viz._density_offsets(values[:, 0], np.random.default_rng(0))

    assert offsets.shape == (N_INSTANCES,)
    assert np.abs(offsets).max() <= 0.8


def test_density_offsets_on_constant_and_empty_input() -> None:
    """Verify degenerate columns produce no vertical spread."""
    rng = np.random.default_rng(0)
    np.testing.assert_array_equal(viz._density_offsets(np.zeros(5), rng), np.zeros(5))
    assert viz._density_offsets(np.array([]), rng).shape == (0,)


def test_pca_2d_returns_two_components(values: np.ndarray) -> None:
    """Verify the projection yields two coordinates per instance."""
    assert viz._pca_2d(values).shape == (N_INSTANCES, 2)


def test_pca_2d_with_single_feature() -> None:
    """Verify a single-column matrix is padded with a zero second component."""
    coords = viz._pca_2d(np.array([[1.0], [2.0], [3.0]]))
    assert coords.shape == (3, 2)
    np.testing.assert_array_equal(coords[:, 1], np.zeros(3))


@pytest.mark.parametrize("strategy", ["hclust", "output", "none"])
def test_order_instances(values: np.ndarray, strategy: str) -> None:
    """Verify every ordering strategy returns a valid permutation."""
    order = viz._order_instances(values, strategy)
    np.testing.assert_array_equal(np.sort(order), np.arange(N_INSTANCES))


def test_order_instances_falls_back_when_too_large() -> None:
    """Verify clustering is skipped for datasets too large to cluster."""
    order = viz._order_instances(np.zeros((2001, 2)), "hclust")
    assert order.shape == (2001,)


def test_feature_index_by_name(feature_names: list[str], values: np.ndarray) -> None:
    """Verify a feature name resolves to its column index."""
    assert viz._feature_index("feat_3", feature_names, values) == 3


def test_feature_index_defaults_to_most_important(
    feature_names: list[str],
) -> None:
    """Verify None selects the feature with the largest mean magnitude."""
    scores = np.array([[0.1, 5.0, 0.2, 0.0, 0.0, 0.0]])
    assert viz._feature_index(None, feature_names, scores) == 1


def test_feature_index_rejects_unknown_name(
    feature_names: list[str], values: np.ndarray
) -> None:
    """Verify an unknown feature name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown feature"):
        viz._feature_index("missing", feature_names, values)


def test_feature_index_rejects_out_of_range(
    feature_names: list[str], values: np.ndarray
) -> None:
    """Verify an out-of-range index raises ValueError."""
    with pytest.raises(ValueError, match="out of range"):
        viz._feature_index(99, feature_names, values)


def test_numeric_data_rejects_mismatched_and_non_numeric() -> None:
    """Verify only numeric data of the expected shape is accepted."""
    assert viz._numeric_data(None, (2, 2)) is None
    assert viz._numeric_data(np.array(["a", "b"]), (1, 2)) is None
    assert viz._numeric_data(np.zeros((3, 3)), (2, 2)) is None
    assert viz._numeric_data(np.zeros((2, 2)), (2, 2)) is not None


def test_significant_splits_detects_a_shift() -> None:
    """Verify a step change in the attribution series is flagged."""
    series = np.concatenate([np.zeros(100), np.full(100, 10.0)])
    assert viz._significant_splits(series, 50)


def test_significant_splits_ignores_short_series() -> None:
    """Verify short series are not tested for drift."""
    assert viz._significant_splits(np.zeros(5), 50) == []


def test_normalise_backgrounds_fallbacks() -> None:
    """Verify unusable backgrounds fall back to a blank canvas."""
    blank = viz._normalise_backgrounds(None, 2, (4, 4))
    assert blank.shape == (2, 4, 4, 3)

    mismatched = viz._normalise_backgrounds(np.zeros((1, 9, 9, 3)), 1, (4, 4))
    assert mismatched.shape == (1, 4, 4, 3)


@pytest.mark.parametrize(
    "source",
    [
        np.full((4, 4), 200.0),  # 2D grayscale, 0-255 range
        np.full((4, 4, 3), 0.5),  # single RGB image
        np.full((2, 4, 4), -0.5),  # grayscale batch in [-1, 1]
        np.full((1, 4, 4, 1), 0.3),  # explicit single channel
    ],
)
def test_normalise_backgrounds_layouts(source: np.ndarray) -> None:
    """Verify assorted image layouts normalise to (n, H, W, 3) in [0, 1]."""
    images = viz._normalise_backgrounds(source, 2, (4, 4))

    assert images.shape == (2, 4, 4, 3)
    assert images.min() >= 0.0
    assert images.max() <= 1.0


def test_prepare_image_arrays_rejects_bad_rank() -> None:
    """Verify a rank the image plot cannot interpret raises ValueError."""
    with pytest.raises(ValueError, match="rank 2 to 5"):
        viz._prepare_image_arrays(Explanation(values=np.zeros(4)), None)


# ==============================================================================
# MATPLOTLIB BACKEND
# ==============================================================================


def test_bar_global(exp_2d: Explanation) -> None:
    """Verify a batched explanation renders a mean-magnitude bar chart."""
    fig = viz.bar(exp_2d, show=False)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert fig.axes[0].get_xlabel() == f"mean(|{viz.VALUE_LABEL}|)"


def test_bar_local_uses_signed_label(exp_1d: Explanation) -> None:
    """Verify a single instance renders signed attributions."""
    fig = viz.bar(exp_1d, show=False, title="Local")
    assert fig is not None
    assert fig.axes[0].get_xlabel() == viz.VALUE_LABEL


def test_bar_groups_minor_features(exp_2d: Explanation) -> None:
    """Verify max_display limits the number of drawn bars."""
    fig = viz.bar(exp_2d, max_display=3, show=False)
    assert fig is not None
    assert len(fig.axes[0].patches) == 3


def test_bar_on_all_zero_values() -> None:
    """Verify an explanation with no signal still produces a valid axis range."""
    fig = viz.bar(Explanation(values=np.zeros((5, 3))), show=False)
    assert fig is not None
    low, high = fig.axes[0].get_xlim()
    assert low < high


def test_bar_into_existing_axes(exp_2d: Explanation) -> None:
    """Verify an existing axes object is reused rather than replaced."""
    _, ax = plt.subplots()
    fig = viz.bar(exp_2d, ax=ax, show=False)
    assert fig is ax.get_figure()


def test_waterfall_bars_sum_to_the_prediction(
    exp_1d: Explanation, values: np.ndarray
) -> None:
    """Verify the waterfall is additive: the bars close the base-to-output gap."""
    fig = viz.waterfall(exp_1d, max_display=None, show=False)
    assert fig is not None

    bars = [p for p in fig.axes[0].patches if isinstance(p, Rectangle)]
    total = sum(bar.get_width() for bar in bars)
    assert total == pytest.approx(values[0].sum())


def test_waterfall_rejects_batched_explanation(exp_2d: Explanation) -> None:
    """Verify a multi-instance explanation is rejected with a clear message."""
    with pytest.raises(ValueError, match="single instance"):
        viz.waterfall(exp_2d, show=False)


def test_waterfall_accepts_wrapped_single_instance(
    values: np.ndarray, feature_names: list[str]
) -> None:
    """Verify a (1, n_features) explanation is unwrapped automatically."""
    exp = Explanation(
        values=values[:1],
        base_values=np.array([0.2]),
        data=values[:1],
        feature_names=feature_names,
    )
    assert viz.waterfall(exp, show=False) is not None


def test_waterfall_with_string_data() -> None:
    """Verify non-numeric feature values are labelled without raising."""
    exp = Explanation(
        values=np.array([0.5, -0.5]),
        data=np.array(["yes", "no"]),
        feature_names=["a", "b"],
    )
    assert viz.waterfall(exp, show=False) is not None


def test_decision_matplotlib(
    values: np.ndarray, data: np.ndarray, feature_names: list[str]
) -> None:
    """Verify the decision plot draws one path per instance."""
    fig = viz.decision(0.4, values, data, feature_names, show=False)
    assert fig is not None
    assert len(fig.axes[0].lines) >= N_INSTANCES


def test_decision_with_single_instance(
    values: np.ndarray, data: np.ndarray, feature_names: list[str]
) -> None:
    """Verify a 1D input is promoted to a single decision path."""
    assert viz.decision(0.4, values[0], data[0], feature_names, show=False) is not None


def test_decision_with_array_base_value(values: np.ndarray) -> None:
    """Verify an array base value is reduced to a scalar."""
    assert viz.decision(np.array([0.4]), values, show=False) is not None


def test_scatter_defaults_to_most_important_feature(exp_2d: Explanation) -> None:
    """Verify the scatter plot labels the axis with the chosen feature."""
    fig = viz.scatter(exp_2d, show=False)
    assert fig is not None
    assert viz.VALUE_LABEL in fig.axes[0].get_ylabel()


def test_scatter_with_named_feature_and_color(exp_2d: Explanation) -> None:
    """Verify features may be selected by name for both position and colour."""
    fig = viz.scatter(exp_2d, ind="feat_2", color="feat_4", show=False)
    assert fig is not None
    assert fig.axes[0].get_xlabel() == "feat_2"


def test_scatter_without_data_uses_instance_index(
    values: np.ndarray, feature_names: list[str]
) -> None:
    """Verify a missing data matrix falls back to the instance index."""
    exp = Explanation(values=values, feature_names=feature_names)
    fig = viz.scatter(exp, color="feat_1", show=False)
    assert fig is not None
    assert fig.axes[0].get_xlabel() == "Instance index"


@pytest.mark.parametrize("instance_order", ["hclust", "output", "none"])
def test_heatmap(exp_2d: Explanation, instance_order: str) -> None:
    """Verify the heatmap renders under every instance ordering."""
    assert viz.heatmap(exp_2d, instance_order=instance_order, show=False) is not None


def test_heatmap_groups_minor_features(exp_2d: Explanation) -> None:
    """Verify max_display collapses the weakest features into one row."""
    fig = viz.heatmap(exp_2d, max_display=3, show=False)
    assert fig is not None
    labels = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    assert any("other features" in label for label in labels)


def test_beeswarm(exp_2d: Explanation) -> None:
    """Verify the beeswarm draws one collection per displayed feature."""
    fig = viz.beeswarm(exp_2d, max_display=4, show=False)
    assert fig is not None
    assert len(fig.axes[0].collections) == 4


def test_beeswarm_without_data(values: np.ndarray) -> None:
    """Verify a missing data matrix falls back to uncoloured points."""
    assert viz.beeswarm(Explanation(values=values), show=False) is not None


def test_violin(exp_2d: Explanation) -> None:
    """Verify the violin summary renders."""
    assert viz.violin(exp_2d, max_display=3, show=False) is not None


def test_violin_with_constant_feature(values: np.ndarray) -> None:
    """Verify a zero-variance feature does not break the density estimate."""
    constant = values.copy()
    constant[:, 0] = 1.0
    assert viz.violin(Explanation(values=constant), show=False) is not None


def test_embedding(exp_2d: Explanation) -> None:
    """Verify the embedding projects to two labelled components."""
    fig = viz.embedding(0, exp_2d, show=False)
    assert fig is not None
    assert fig.axes[0].get_xlabel() == "Component 1"


def test_group_difference(exp_2d: Explanation, data: np.ndarray) -> None:
    """Verify the group difference renders as a bar chart."""
    assert viz.group_difference(exp_2d, data[:, 0] > 0, show=False) is not None


def test_group_difference_rejects_mismatched_mask(exp_2d: Explanation) -> None:
    """Verify a mask of the wrong length raises ValueError."""
    with pytest.raises(ValueError, match="but the explanation holds"):
        viz.group_difference(exp_2d, np.ones(3, dtype=bool), show=False)


@pytest.mark.parametrize("mask_value", [True, False])
def test_group_difference_rejects_degenerate_mask(
    exp_2d: Explanation, mask_value: bool
) -> None:
    """Verify an all-true or all-false mask raises ValueError."""
    mask = np.full(N_INSTANCES, mask_value, dtype=bool)
    with pytest.raises(ValueError, match="proper subset"):
        viz.group_difference(exp_2d, mask, show=False)


def test_monitoring(exp_2d: Explanation, data: np.ndarray) -> None:
    """Verify the monitoring plot renders with feature colouring."""
    fig = viz.monitoring(0, exp_2d, data, show=False)
    assert fig is not None
    assert fig.axes[0].get_xlabel() == "Instance index"


def test_monitoring_marks_detected_drift(feature_names: list[str]) -> None:
    """Verify a distribution shift is marked with a vertical rule."""
    drifting = np.zeros((200, N_FEATURES))
    drifting[100:, 0] = 10.0
    exp = Explanation(values=drifting, feature_names=feature_names)

    fig = viz.monitoring(0, exp, None, show=False)
    assert fig is not None
    assert len(fig.axes[0].lines) > 1


def test_image(exp_image: Explanation) -> None:
    """Verify the image plot draws an input column plus one column per output."""
    fig = viz.image(exp_image, show=False)
    assert fig is not None
    assert len(fig.axes) >= 4


def test_image_with_explicit_pixels_and_labels(
    exp_image: Explanation, rng: np.random.Generator
) -> None:
    """Verify explicit background pixels and column labels are honoured."""
    pixels = rng.random((2, 12, 12, 3))
    assert viz.image(exp_image, pixels, labels=["cat"], show=False) is not None


@pytest.mark.parametrize(
    "shape", [(12, 12), (2, 12, 12), (2, 12, 12, 3), (2, 12, 12, 3, 2)]
)
def test_image_accepts_assorted_ranks(shape: tuple[int, ...]) -> None:
    """Verify every supported attribution rank renders."""
    exp = Explanation(values=np.random.default_rng(0).normal(size=shape))
    assert viz.image(exp, show=False) is not None


def test_image_to_text(exp_multimodal: Explanation) -> None:
    """Verify the multimodal plot renders one panel per output token."""
    assert viz.image_to_text(exp_multimodal, show=False) is not None


def test_image_to_text_rejects_low_rank(exp_image: Explanation) -> None:
    """Verify a non-multimodal explanation raises ValueError."""
    with pytest.raises(ValueError, match="5D explanations"):
        viz.image_to_text(exp_image, show=False)


def test_partial_dependence(data: np.ndarray, feature_names: list[str]) -> None:
    """Verify the partial dependence curve is drawn on top of the ICE lines."""
    fig = viz.partial_dependence(
        0, lambda x: x[:, 0] ** 2, data, feature_names=feature_names, show=False
    )
    assert fig is not None
    assert fig.axes[0].get_xlabel() == "feat_0"


def test_partial_dependence_without_ice(data: np.ndarray) -> None:
    """Verify ICE lines can be disabled."""
    fig = viz.partial_dependence(
        0, lambda x: x.sum(axis=1), data, ice=False, show=False
    )
    assert fig is not None
    assert len(fig.axes[0].lines) == 1


def test_partial_dependence_subsamples_ice_lines(data: np.ndarray) -> None:
    """Verify the ICE line count is capped."""
    fig = viz.partial_dependence(
        0, lambda x: x.sum(axis=1), data, max_ice_lines=5, show=False
    )
    assert fig is not None
    assert len(fig.axes[0].lines) == 6  # 5 ICE lines plus the average


def test_partial_dependence_on_low_cardinality_feature() -> None:
    """Verify a categorical-like feature uses its distinct values as the grid."""
    categorical = np.column_stack([np.repeat([0.0, 1.0, 2.0], 5), np.zeros(15)])
    fig = viz.partial_dependence(0, lambda x: x[:, 0], categorical, show=False)
    assert fig is not None
    assert np.asarray(fig.axes[0].lines[-1].get_xdata()).shape == (3,)


def test_initjs_is_a_noop() -> None:
    """Verify initjs exists for compatibility and draws nothing."""
    viz.initjs()

    assert plt.get_fignums() == []


# ==============================================================================
# PLOTLY BACKEND
# ==============================================================================


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("bar", ()),
        ("scatter", ()),
        ("heatmap", ()),
        ("beeswarm", ()),
        ("violin", ()),
    ],
)
def test_plotly_backend_batched(
    exp_2d: Explanation, name: str, args: tuple[Any, ...]
) -> None:
    """Verify each batched plot returns a plotly figure."""
    fig = getattr(viz, name)(exp_2d, *args, backend="plotly", show=False)
    assert isinstance(fig, go.Figure)
    assert fig.data


def test_plotly_waterfall(exp_1d: Explanation) -> None:
    """Verify the plotly waterfall returns a populated figure."""
    fig = viz.waterfall(exp_1d, backend="plotly", show=False)
    assert isinstance(fig, go.Figure)


def test_plotly_embedding(exp_2d: Explanation) -> None:
    """Verify the plotly embedding returns a populated figure."""
    assert isinstance(viz.embedding(0, exp_2d, backend="plotly", show=False), go.Figure)


def test_plotly_decision(values: np.ndarray, feature_names: list[str]) -> None:
    """Verify the plotly decision plot draws one trace per instance."""
    fig = viz.decision(
        0.4, values, feature_names=feature_names, backend="plotly", show=False
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == N_INSTANCES


def test_plotly_monitoring(exp_2d: Explanation, data: np.ndarray) -> None:
    """Verify the plotly monitoring plot returns a figure."""
    assert isinstance(
        viz.monitoring(0, exp_2d, data, backend="plotly", show=False), go.Figure
    )


def test_plotly_beeswarm_without_data(values: np.ndarray) -> None:
    """Verify the plotly beeswarm copes with a missing data matrix."""
    exp = Explanation(values=values)
    assert isinstance(viz.beeswarm(exp, backend="plotly", show=False), go.Figure)


def test_plotly_partial_dependence(data: np.ndarray) -> None:
    """Verify the plotly partial dependence plot returns a figure."""
    fig = viz.partial_dependence(
        0, lambda x: x.sum(axis=1), data, backend="plotly", show=False
    )
    assert isinstance(fig, go.Figure)


def test_plotly_partial_dependence_without_ice(data: np.ndarray) -> None:
    """Verify disabling ICE leaves only the average curve."""
    fig = viz.partial_dependence(
        0, lambda x: x.sum(axis=1), data, ice=False, backend="plotly", show=False
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1


# ==============================================================================
# HTML OUTPUT (JAVASCRIPT-FREE)
# ==============================================================================


def test_text_returns_html(exp_1d: Explanation) -> None:
    """Verify the text plot returns HTML containing every token."""
    html = viz.text(exp_1d, show=False, title="Tokens")

    assert "<span" in html
    assert "Tokens" in html
    for name in ("feat_0", "feat_1"):
        assert name in html


def test_text_escapes_markup() -> None:
    """Verify tokens are HTML-escaped so they cannot inject markup."""
    exp = Explanation(
        values=np.array([1.0]), feature_names=["<script>alert(1)</script>"]
    )
    html = viz.text(exp, show=False)

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


@pytest.mark.parametrize("plot", ["text", "force"])
def test_html_output_contains_no_javascript(exp_1d: Explanation, plot: str) -> None:
    """Verify the HTML plots are static: no scripts, handlers or remote assets."""
    html = (
        viz.text(exp_1d, show=False)
        if plot == "text"
        else viz.force(exp_1d, backend="html", show=False)
    )
    lowered = str(html).lower()

    assert "<script" not in lowered
    assert "onclick" not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered


def test_force_html_reports_base_and_prediction(exp_1d: Explanation) -> None:
    """Verify the force HTML states both the expected value and the output."""
    html = str(viz.force(exp_1d, backend="html", show=False, title="Force"))

    assert "E[f(X)]" in html
    assert "f(x)" in html
    assert "Force" in html


def test_force_matplotlib(exp_1d: Explanation) -> None:
    """Verify the matplotlib force layout renders."""
    fig = viz.force(exp_1d, show=False, title="Force")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_force_rejects_plotly_backend(exp_1d: Explanation) -> None:
    """Verify the force plot advertises only its supported backends."""
    with pytest.raises(ValueError, match="not supported by this plot"):
        viz.force(exp_1d, backend="plotly", show=False)


def test_wrap_html_document_escapes_title() -> None:
    """Verify the standalone document wrapper escapes its title."""
    document = viz._wrap_html_document("<p>body</p>", title="<b>t</b>")

    assert document.startswith("<!doctype html>")
    assert "&lt;b&gt;t&lt;/b&gt;" in document


def test_display_html_without_ipython() -> None:
    """Verify HTML display degrades gracefully when IPython is unavailable."""
    with patch.dict("sys.modules", {"IPython.display": None}):
        viz._display_html("<p>x</p>")  # must not raise


# ==============================================================================
# OUTPUT ROUTING (show / save_path)
# ==============================================================================


def test_matplotlib_show_path(exp_2d: Explanation) -> None:
    """Verify show=True renders and returns nothing."""
    with patch("matplotlib.pyplot.show") as mock_show:
        assert viz.bar(exp_2d, show=True) is None
    mock_show.assert_called_once()


def test_matplotlib_save_path(exp_2d: Explanation, tmp_path: Path) -> None:
    """Verify save_path writes a file and suppresses rendering."""
    destination = tmp_path / "bar.png"
    with patch("matplotlib.pyplot.show") as mock_show:
        assert viz.bar(exp_2d, save_path=destination) is None

    assert destination.exists()
    mock_show.assert_not_called()


def test_plotly_show_path(exp_2d: Explanation) -> None:
    """Verify show=True on the plotly backend renders and returns nothing."""
    with patch.object(go.Figure, "show") as mock_show:
        assert viz.bar(exp_2d, backend="plotly", show=True) is None
    mock_show.assert_called_once()


def test_plotly_save_html(exp_2d: Explanation, tmp_path: Path) -> None:
    """Verify a .html save path writes an interactive document."""
    destination = tmp_path / "bar.html"
    assert viz.bar(exp_2d, backend="plotly", save_path=destination) is None
    assert destination.exists()


def test_plotly_save_static_image(exp_2d: Explanation, tmp_path: Path) -> None:
    """Verify a non-HTML save path routes to the static image writer."""
    with patch.object(go.Figure, "write_image") as mock_write:
        assert viz.bar(exp_2d, backend="plotly", save_path=tmp_path / "bar.png") is None
    mock_write.assert_called_once()


def test_html_save_path_writes_document(exp_1d: Explanation, tmp_path: Path) -> None:
    """Verify the HTML plots write a standalone document and return the fragment."""
    destination = tmp_path / "text.html"
    html = viz.text(exp_1d, save_path=destination)

    assert destination.exists()
    assert destination.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert html in destination.read_text(encoding="utf-8")


def test_html_show_path_displays(exp_1d: Explanation) -> None:
    """Verify show=True routes the fragment to the notebook display hook."""
    with patch.object(viz, "_display_html") as mock_display:
        viz.text(exp_1d, show=True)
    mock_display.assert_called_once()
