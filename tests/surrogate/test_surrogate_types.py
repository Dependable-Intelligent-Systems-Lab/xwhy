"""Unit tests for surrogate types."""

import pytest

from xwhy.surrogate.types import SurrogateType


def test_surrogate_type_from_str_success() -> None:
    """Test successful conversion from valid strings and enum instances."""
    assert SurrogateType.from_str("glm_ols") == SurrogateType.GLM_OLS
    assert SurrogateType.from_str("lime_ridge") == SurrogateType.LIME_RIDGE
    assert SurrogateType.from_str("randomforest") == SurrogateType.RANDOMFOREST

    assert SurrogateType.from_str(SurrogateType.BAYLIME) == SurrogateType.BAYLIME


def test_surrogate_type_from_str_invalid() -> None:
    """Verify that invalid input raises a descriptive ValueError."""
    invalid_input = "invalid_surrogate"

    with pytest.raises(
        ValueError, match=f"'{invalid_input}' is not a valid SurrogateType"
    ):
        SurrogateType.from_str(invalid_input)


"""Unit tests for the surrogate model types enumeration."""


def test_is_linear_model_property() -> None:
    """Verify that the is_linear_model property correctly identifies linear models.

    Ensures that all GLM, LIME, and BAYLIME variants return True, while
    tree-based models return False.
    """
    assert SurrogateType.GLM_OLS.is_linear_model is True
    assert SurrogateType.GLM_RIDGE.is_linear_model is True
    assert SurrogateType.LIME.is_linear_model is True
    assert SurrogateType.LIME_RIDGE.is_linear_model is True
    assert SurrogateType.BAYLIME.is_linear_model is True

    assert SurrogateType.RANDOMFOREST.is_linear_model is False
    assert SurrogateType.GRADIENT_BOOSTING.is_linear_model is False
    assert SurrogateType.XGBOOST.is_linear_model is False


def test_is_tree_model_property() -> None:
    """Verify that the is_tree_model property correctly identifies tree models.

    Ensures that Random Forest, Gradient Boosting, and XGBoost return True,
    while linear models return False.
    """
    assert SurrogateType.RANDOMFOREST.is_tree_model is True
    assert SurrogateType.GRADIENT_BOOSTING.is_tree_model is True
    assert SurrogateType.XGBOOST.is_tree_model is True

    assert SurrogateType.GLM_OLS.is_tree_model is False
    assert SurrogateType.GLM_RIDGE.is_tree_model is False
    assert SurrogateType.LIME.is_tree_model is False
    assert SurrogateType.LIME_RIDGE.is_tree_model is False
    assert SurrogateType.BAYLIME.is_tree_model is False
