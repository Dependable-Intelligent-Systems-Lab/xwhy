"""Surrogate model types definitions."""

from __future__ import annotations

from enum import StrEnum


class SurrogateType(StrEnum):
    """Enumeration for supported surrogate model types."""

    # Linear Models
    GLM_OLS = "glm_ols"  # Global/Unweighted Ordinary Least Squares (OLS)
    GLM_RIDGE = "glm_ridge"  # Global/Unweighted Ridge Regression (L2)
    LIME = "lime_ols"  # Local Weighted OLS
    LIME_RIDGE = "lime_ridge"  # Local Weighted Ridge Regression (L2)
    BAYLIME = "baylime"  # Bayesian Weighted Ridge (L2)
    # Non-Linear Tree-Based Models
    RANDOMFOREST = "randomforest"
    GRADIENT_BOOSTING = "gradientboosting"
    XGBOOST = "xgboost"

    @classmethod
    def from_str(cls, value: str | SurrogateType) -> SurrogateType:
        """Safely convert a string or enum instance to SurrogateType."""
        try:
            return cls(value)
        except ValueError as err:
            valid_options = [item.value for item in cls]
            raise ValueError(
                f"'{value}' is not a valid SurrogateType. "
                f"Please choose from: {valid_options}"
            ) from err
