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
