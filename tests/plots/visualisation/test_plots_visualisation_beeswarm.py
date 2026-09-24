"""Test beeswarm module."""

import matplotlib

matplotlib.use("Agg")

from collections.abc import Generator
from unittest.mock import patch

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from xwhy.plots.visualisation.base import BLUE, DimensionError, Explanation
from xwhy.plots.visualisation.beeswarm import (
    _shap_beeswarm,
    beeswarm,
    convert_color,
    convert_ordering,
    fill_counts,
    fill_internal_max_values,
    get_sort_order,
    merge_nodes,
    safe_isinstance,
    sort_inds,
)

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


@pytest.fixture(autouse=True)
def _cleanup_plots() -> Generator[None, None, None]:
    yield
    plt.close("all")


def test_safe_isinstance() -> None:
    """Test safe_isinstance."""
    assert safe_isinstance(cm.viridis, "matplotlib.colors.Colormap")
    assert safe_isinstance(cm.viridis, ["SomethingElse", "Colormap"])
    assert not safe_isinstance(cm.viridis, "NonExistent")


def test_convert_color() -> None:
    """Test convert_color."""
    # ndarray
    arr = np.array([1, 0, 0, 1])
    assert convert_color(arr) is arr
    # string alias
    assert convert_color("shap_red") == "#FF0D57"
    assert convert_color("shap_blue") == BLUE
    # Colormap object
    cmap = cm.viridis
    assert convert_color(cmap) is cmap
    # Existing colormap string
    assert convert_color("viridis") != "viridis"  # returns a Colormap
    # Non-existent colormap string fallback to string
    assert convert_color("some_magic_color") == "some_magic_color"


def test_fill_internal_max_values() -> None:
    """Test fill_internal_max_values."""
    # 4 leaves -> 3 merges
    # merges: [left, right, dist, max_val]
    # n_leaves = 4
    partition_tree = np.array(
        [
            [0, 1, 0.1, 0.0],
            [2, 3, 0.2, 0.0],
            [4, 5, 0.3, 0.0],
        ]
    )
    leaf_values = np.array([1.0, -2.0, 3.0, 4.0])
    new_tree = fill_internal_max_values(partition_tree, leaf_values)
    assert new_tree[0, 3] == 2.0  # max(|1|, |-2|)
    assert new_tree[1, 3] == 4.0  # max(|3|, |4|)
    assert new_tree[2, 3] == 4.0  # max(2, 4)


def test_fill_counts() -> None:
    """Test fill_counts."""
    partition_tree = np.array(
        [
            [0, 1, 0.1, 0.0],
            [2, 3, 0.2, 0.0],
            [4, 5, 0.3, 0.0],
        ]
    )
    fill_counts(partition_tree)
    assert partition_tree[0, 3] == 2.0
    assert partition_tree[1, 3] == 2.0
    assert partition_tree[2, 3] == 4.0


def test_sort_inds() -> None:
    """Test sort_inds."""
    partition_tree = np.array(
        [
            [0, 1, 0.1, 0.0],
            [2, 3, 0.2, 0.0],
            [4, 5, 0.3, 0.0],
        ]
    )

    leaf_values = np.array([1.0, -2.0, 3.0, 4.0])
    inds = sort_inds(partition_tree, leaf_values)
    # The larger value is visited first
    # max of tree[4] vs tree[5] -> tree[1] is 4.0, tree[0] is 2.0. So 1 first, then 0.
    # Inside 1 (leaves 2,3), leaf 3 is 4.0, leaf 2 is 3.0. So 3 then 2.
    # Inside 0 (leaves 0,1), leaf 1 is 2.0, leaf 0 is 1.0. So 1 then 0.
    assert inds == [3, 2, 0, 1]


def test_convert_ordering() -> None:
    """Test convert_ordering."""

    class OpChain:
        def __init__(self, name: str = "op") -> None:
            self.name = name

        def apply(self, exp: object) -> object:
            return np.array([1, 0, 2])

    op = OpChain()
    shap_values = np.array([[1, 2, 3]])
    assert np.array_equal(convert_ordering(op, shap_values), [1, 0, 2])

    class FakeFlip:
        values = np.array([2, 1, 0])

    class FakeArgsort:
        flip = FakeFlip()

    exp_type_argsort = type(
        "Explanation",
        (object,),
        {"op_history": [OpChain("argsort")], "values": np.array([0, 1, 2])},
    )

    assert np.array_equal(convert_ordering(exp_type_argsort(), shap_values), [0, 1, 2])

    exp_type_no_history = type(
        "Explanation",
        (object,),
        {"op_history": [OpChain("other")], "argsort": FakeArgsort()},
    )

    assert np.array_equal(
        convert_ordering(exp_type_no_history(), shap_values), [2, 1, 0]
    )

    # Raw array
    assert np.array_equal(convert_ordering([1, 2, 3], shap_values), [1, 2, 3])


def test_get_sort_order() -> None:
    """Test get_sort_order."""
    dist = np.array(
        [
            [0, 0.2, 0.8],
            [0.2, 0, 0.9],
            [0.8, 0.9, 0],
        ]
    )
    clust_order = [0, 1, 2]
    feature_order = np.array([2, 0, 1])
    # Threshold 0.5: 0 and 1 are close, 2 is far.
    # 2 is first, dist[2,0]=0.8 > threshold. So it looks for closest in cluster_order.
    # Logic tries to put things close in cluster order if dist < threshold.
    new_order = get_sort_order(dist, clust_order, 0.5, feature_order)
    # 2, 0, 1 -> dist[2,0] > 0.5, dist[2,1] > 0.5 -> no reordering with 2.
    # then 0, 1 -> dist[0,1] < 0.5, and they are ordered 0 then 1 in clust_order.
    assert isinstance(new_order, np.ndarray)


def test_merge_nodes() -> None:
    """Test merge_nodes."""
    partition_tree = np.array(
        [
            [0, 1, 0.1, 0.0],
            [2, 3, 0.2, 0.0],
            [4, 5, 0.3, 0.0],
        ]
    )

    values = np.array([0.1, 0.2, 3.0, 4.0])

    # leaves 0 and 1 have lowest sum: 0.1 + 0.2 = 0.3

    # They should be merged.

    _new_tree, ind1, ind2 = merge_nodes(values, partition_tree)

    assert ind1 == 0

    assert ind2 == 1

    assert _new_tree.shape == (2, 4)


def test_shap_beeswarm_invalid_args() -> None:
    """Test invalid arguments to _shap_beeswarm."""
    with pytest.raises(TypeError, match="requires an `Explanation` object"):
        _shap_beeswarm("not_explanation")

    exp1d = Explanation(values=np.array([1, 2]))

    with pytest.raises(ValueError, match="does not support plotting a single instance"):
        _shap_beeswarm(exp1d)

    exp3d = Explanation(values=np.array([[[1]]]))

    with pytest.raises(ValueError, match="more than one dimension"):
        _shap_beeswarm(exp3d)

    exp = Explanation(values=np.array([[1, 2]]))

    _fig, ax = plt.subplots()

    with pytest.raises(
        ValueError, match="does not support passing an axis and adjusting"
    ):
        _shap_beeswarm(exp, ax=ax, plot_size=5)


def test_shap_beeswarm_basic() -> None:
    """Test _shap_beeswarm basic execution."""
    exp = Explanation(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        data=np.array([[10, 20], [30, 40]]),
        feature_names=["A", "B"],
    )

    ax = _shap_beeswarm(exp, show=False)

    assert ax is not None

    assert len(ax.collections) > 0  # Should have scatter points


def test_shap_beeswarm_sparse_data() -> None:
    """Test _shap_beeswarm with sparse data."""
    exp = Explanation(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        data=scipy.sparse.csr_matrix(np.array([[10, 20], [30, 40]])),
        feature_names=["A", "B"],
    )

    ax = _shap_beeswarm(exp, show=False)

    assert ax is not None


def test_shap_beeswarm_pandas_data() -> None:
    """Test _shap_beeswarm with pandas dataframe."""
    exp = Explanation(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        data=pd.DataFrame({"A": [10, 30], "B": ["cat", "dog"]}),
    )

    ax = _shap_beeswarm(exp, show=False)

    assert ax is not None


def test_shap_beeswarm_list_data() -> None:
    """Test _shap_beeswarm with list data representing names."""
    exp = Explanation(
        values=np.array([[1.0, 2.0]]),
        data=["FeatA", "FeatB"],
    )

    ax = _shap_beeswarm(exp, show=False)

    assert ax is not None


def test_shap_beeswarm_dimension_error() -> None:
    """Test dimension error."""
    exp = Explanation(
        values=np.array([[1.0, 2.0]]),
        data=np.array([[10, 20, 30]]),  # Mismatch
    )

    with pytest.raises(DimensionError):
        _shap_beeswarm(exp, show=False)

    exp2 = Explanation(
        values=np.array([[1.0, 2.0, 3.0]]),
        data=np.array([[10, 20]]),  # Off by one, possible offset
    )

    with pytest.raises(DimensionError, match="constant offset"):
        _shap_beeswarm(exp2, show=False)


def test_shap_beeswarm_clustering() -> None:
    """Test clustering logic."""
    exp = Explanation(
        values=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        data=np.array([[1, 2, 3], [4, 5, 6]]),
        clustering=np.array([[0, 1, 0.1, 2.0], [3, 2, 0.2, 3.0]]),
    )

    ax = _shap_beeswarm(exp, clustering=None, show=False, max_display=2)

    assert ax is not None

    with pytest.raises(ValueError, match="not seem to be a partition tree"):
        _shap_beeswarm(exp, clustering=np.array([[1, 2]]), show=False)


def test_shap_beeswarm_various_options() -> None:
    """Test various options like log_scale, color_bar, plot_size."""
    exp = Explanation(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )

    ax = _shap_beeswarm(
        exp,
        show=False,
        log_scale=True,
        color_bar=False,
        plot_size=(10, 5),
        color="viridis",
    )

    assert ax is not None

    assert ax.get_xscale() == "symlog"

    ax2 = _shap_beeswarm(exp, show=False, plot_size=5.0)

    assert ax2 is not None


def test_shap_beeswarm_categorical_color() -> None:
    """Test categorical feature coloring fallback."""
    exp = Explanation(
        values=np.array([[1.0], [2.0]]),
        data=pd.DataFrame({"A": ["cat1", "cat2"]}),
    )

    ax = _shap_beeswarm(exp, show=False, color="viridis")

    assert ax is not None


def test_shap_beeswarm_large_data() -> None:
    """Test rasterization branch for large data."""
    exp = Explanation(
        values=np.random.default_rng(42).standard_normal((600, 1)),
        data=np.random.default_rng(42).standard_normal((600, 1)),
    )

    ax = _shap_beeswarm(exp, show=False)

    assert ax is not None


def test_beeswarm_wrapper() -> None:
    """Test beeswarm wrapper."""
    exp = Explanation(values=np.array([[1.0, 2.0], [3.0, 4.0]]))

    # Return figure

    fig = beeswarm(exp, show=False, figsize=(10, 5))

    assert fig is not None

    # With title

    fig2 = beeswarm(exp, show=False, title="My Title")

    assert fig2 is not None

    # Mock plt.show

    with patch("matplotlib.pyplot.show") as mock_show:
        res = beeswarm(exp, show=True)

        mock_show.assert_called_once()

        assert res is None


def test_get_sort_order_swaps() -> None:
    """Test get_sort_order swaps."""
    dist = np.array(
        [
            [0.0, 0.2, 0.4, 0.8],
            [0.2, 0.0, 0.6, 0.9],
            [0.4, 0.6, 0.0, 0.3],
            [0.8, 0.9, 0.3, 0.0],
        ]
    )

    clust_order = [0, 2, 1, 3]

    feature_order = np.array([3, 0, 1, 2])

    # threshold 0.5

    # i=0, ind1=3. next_ind=0.

    #   j=1 (ind2=0) -> dist[3,0]=0.8 > threshold

    #   j=2 (ind2=1) -> dist[3,1]=0.9 > threshold

    #   j=3 (ind2=2) -> dist[3,2]=0.3 <= threshold.

    #       dist[3,0]=0.8 > threshold -> true! next_ind=2. next_ind_pos=3

    new_order = get_sort_order(dist, clust_order, 0.5, feature_order)

    assert len(new_order) == 4


def test_merge_nodes_swap_inds() -> None:
    """Test merge_nodes when ind1 > ind2."""
    partition_tree = np.array(
        [
            [1, 0, 0.1, 0.0],
            [2, 3, 0.2, 0.0],
            [4, 5, 0.3, 0.0],
        ]
    )

    values = np.array([0.1, 0.2, 3.0, 4.0])

    _new_tree, ind1, ind2 = merge_nodes(values, partition_tree)

    assert ind1 == 0

    assert ind2 == 1


def test_shap_beeswarm_order_none() -> None:
    """Test _shap_beeswarm with order=None."""
    exp = Explanation(values=np.array([[1.0, 2.0], [3.0, 4.0]]))

    _shap_beeswarm(exp, show=False, order=None)


def test_shap_beeswarm_features_conditions() -> None:
    """Test features handling conditions."""
    # pd.DataFrame without feature_names

    exp = Explanation(
        values=np.array([[1.0], [2.0]]),
        data=pd.DataFrame({"A": [10, 20]}),
    )

    # df with feature_names None

    exp.feature_names = None

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        _shap_beeswarm(exp, show=False)

    # list data

    exp2 = Explanation(
        values=np.array([[1.0], [2.0]]),
        data=["F1"],
    )

    exp2.feature_names = None

    _shap_beeswarm(exp2, show=False)

    # 1D array data

    exp3 = Explanation(
        values=np.array([[1.0], [2.0]]),
        data=np.array([1, 2]),
    )

    exp3.feature_names = None

    _shap_beeswarm(exp3, show=False)


def test_shap_beeswarm_fig_type_error() -> None:
    """Test figure type error."""
    exp = Explanation(values=np.array([[1.0, 2.0], [3.0, 4.0]]))

    _fig, ax = plt.subplots()

    with (
        patch("matplotlib.axes.Axes.get_figure", return_value="not_a_fig"),
        pytest.raises(TypeError, match="Expected a matplotlib Figure"),
    ):
        _shap_beeswarm(exp, show=False, ax=ax, plot_size=None)


def test_shap_beeswarm_clustering_variance() -> None:
    """Test clustering variance zero."""
    exp = Explanation(values=np.array([[1.0], [2.0]]), clustering=np.zeros((1, 1, 4)))

    with patch("xwhy.plots.visualisation.beeswarm.sort_inds", return_value=[0, 1]):
        _shap_beeswarm(exp, show=False, clustering=None)


def test_shap_beeswarm_clustering_false() -> None:
    """Test clustering=False."""
    exp = Explanation(values=np.array([[1.0, 2.0], [3.0, 4.0]]))

    _shap_beeswarm(exp, show=False, clustering=False)


def test_shap_beeswarm_partition_tree_while_loop() -> None:
    """Test the merge loop logic."""
    exp = Explanation(
        values=np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        feature_names=["f1", "f2", "f3", "f4"],
        clustering=np.array(
            [[[0.0, 1.0, 0.1, 2.0], [2.0, 3.0, 0.2, 2.0], [4.0, 5.0, 0.3, 4.0]]]
        ),
    )

    with (
        patch("xwhy.plots.visualisation.beeswarm.sort_inds", return_value=[0, 1, 2, 3]),
        patch(
            "xwhy.plots.visualisation.beeswarm.merge_nodes", return_value=(None, 0, 1)
        ),
        patch(
            "xwhy.plots.visualisation.beeswarm.convert_ordering",
            side_effect=[
                np.array([3, 2, 1, 0]),
                np.array([3, 2, 1, 0]),
                np.array([1, 0]),
            ],
        ),
    ):
        _shap_beeswarm(exp, show=False, max_display=2, cluster_threshold=100.0)


def test_shap_beeswarm_categorical_exception() -> None:
    """Test categorical fallback exception."""
    exp = Explanation(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        data=np.array([["A", "B"], ["C", "D"]]),
    )

    # The string data will throw exception inside np.array(fvalues, dtype=np.float64)

    # and trigger colored_feature = False

    _shap_beeswarm(exp, show=False)


def test_shap_beeswarm_color_scale_vmin_vmax() -> None:
    """Test color scale where vmin >= vmax."""
    exp = Explanation(
        values=np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]),
        data=np.array([[5.0], [5.0], [5.0], [5.0], [5.0], [5.0]]),
    )

    # vmin will equal vmax, causing it to fall back to min/max

    _shap_beeswarm(exp, show=False)


def test_shap_beeswarm_dimension_error_shaps() -> None:
    """Test dimension error during color scale."""
    exp = Explanation(
        values=np.array([[1.0]]),
        data=np.array([[1.0], [2.0]]),
    )

    # columns match, rows mismatch

    with pytest.raises(DimensionError, match="same number of rows"):
        _shap_beeswarm(exp, show=False)


def test_beeswarm_unreachable_returns() -> None:
    """Test unreachable returns when _finish_matplotlib is None."""
    exp = Explanation(values=np.array([[1.0, 2.0], [3.0, 4.0]]))

    with patch("xwhy.plots.visualisation.beeswarm._finish_matplotlib", None):
        fig = beeswarm(exp, show=False)

        assert fig is not None

        with patch("matplotlib.pyplot.show"):
            res = beeswarm(exp, show=True)

            assert res is None


def test_shap_beeswarm_order_not_none() -> None:
    """Test _shap_beeswarm with explicit order."""
    exp = Explanation(values=np.array([[1.0, 2.0], [3.0, 4.0]]))

    _shap_beeswarm(exp, show=False, order=np.array([1, 0]))


def test_shap_beeswarm_pd_dataframe_no_feature_names() -> None:
    """Test _shap_beeswarm with pd.DataFrame and no feature_names."""
    exp = Explanation(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        data=pd.DataFrame({"A": [10, 20], "B": [30, 40]}),
    )

    _shap_beeswarm(exp, show=False)


def test_shap_beeswarm_list_features_no_feature_names() -> None:
    """Test _shap_beeswarm with list features and no feature_names."""
    exp = Explanation(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        data=[["a", "b"], ["c", "d"]],
    )

    exp.feature_names = None  # Force it to None to cover the branch

    _shap_beeswarm(exp, show=False)


def test_shap_beeswarm_max_display_none() -> None:
    """Test max_display=None."""
    exp = Explanation(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
    )

    _shap_beeswarm(exp, show=False, max_display=None)


def test_shap_beeswarm_vmin_vmax_equal() -> None:
    """Test vmin == vmax branch."""
    exp = Explanation(
        values=np.array([[1.0], [2.0]]),
        data=np.array([[5.0], [5.0]]),
    )

    _shap_beeswarm(exp, show=False)


def test_shap_beeswarm_three_merged_nodes() -> None:
    """Test else branch for len(inds) > 2."""
    exp = Explanation(
        values=np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]),
        data=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
        feature_names=["f1", "f2", "f3", "f4"],
        clustering=np.array(
            [[[0.0, 1.0, 0.1, 2.0], [2.0, 3.0, 0.2, 2.0], [4.0, 5.0, 0.3, 4.0]]]
        ),
    )

    def mock_merge_nodes(val: object, tree: object) -> object:

        return None, 0, 1

    with (
        patch(
            "xwhy.plots.visualisation.beeswarm.sort_inds",
            side_effect=[[0, 1, 2, 3], [0, 1, 2]],
        ),
        patch(
            "xwhy.plots.visualisation.beeswarm.merge_nodes",
            side_effect=mock_merge_nodes,
        ),
        patch(
            "xwhy.plots.visualisation.beeswarm.convert_ordering",
            side_effect=[
                np.array([3, 2, 1, 0]),
                np.array([3, 2, 1, 0]),
                np.array([1, 0]),
            ],
        ),
    ):
        _shap_beeswarm(exp, show=False, max_display=1, cluster_threshold=100.0)


def test_shap_beeswarm_idx2cat() -> None:
    """Test idx2cat branch with pandas dataframe."""
    exp = Explanation(
        values=np.array([[1.0], [2.0]]),
        data=pd.DataFrame({"A": pd.Series(["a", "b"], dtype="category")}),
        feature_names=["f1"],
    )

    _shap_beeswarm(exp, show=False)


def test_shap_beeswarm_color_string() -> None:
    """Test string color."""
    exp = Explanation(
        values=np.array([[1.0], [2.0]]),
    )

    _shap_beeswarm(exp, show=False, color="#000000")


def test_shap_beeswarm_vmin_vmax_percentiles() -> None:
    """Test vmin == vmax percentiles branches."""
    # 5th and 95th are equal, but 1st and 99th are not!

    data = np.full((100, 1), 5.0)

    data[0, 0] = 0.0

    data[99, 0] = 10.0

    exp = Explanation(
        values=np.random.default_rng(0).standard_normal((100, 1)),
        data=data,
    )

    _shap_beeswarm(exp, show=False)


def test_shap_beeswarm_vmin_vmax_greater() -> None:
    """Test vmin > vmax branch."""
    exp = Explanation(
        values=np.array([[1.0], [2.0]]),
        data=np.array([[5.0], [5.0]]),
    )

    with patch(
        "xwhy.plots.visualisation.beeswarm.np.nanpercentile", side_effect=[10.0, 5.0]
    ):
        _shap_beeswarm(exp, show=False)


def test_shap_beeswarm_list_features_with_feature_names() -> None:
    """Test list features where feature_names is NOT None."""
    exp = Explanation(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        data=[["a", "b"], ["c", "d"]],
        feature_names=["f1", "f2"],
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        _shap_beeswarm(exp, show=False)


def test_shap_beeswarm_two_merged_nodes_single_row() -> None:
    """Test len(inds) == 2 branch by passing a single row."""
    exp = Explanation(
        values=np.array([[1.0, 2.0]]),
        data=np.array([[1, 2]]),
        feature_names=["f1", "f2"],
        clustering=np.array([[[0.0, 1.0, 0.1, 2.0]]]),
    )

    def mock_merge_nodes(val: object, tree: object) -> object:
        return None, 0, 1

    with (
        patch("xwhy.plots.visualisation.beeswarm.sort_inds", return_value=[0, 1]),
        patch(
            "xwhy.plots.visualisation.beeswarm.merge_nodes",
            side_effect=mock_merge_nodes,
        ),
        patch(
            "xwhy.plots.visualisation.beeswarm.convert_ordering",
            side_effect=[
                np.array([1, 0]),
                np.array([1, 0]),
                np.array([0]),
            ],
        ),
    ):
        _shap_beeswarm(exp, show=False, max_display=1, cluster_threshold=100.0)
