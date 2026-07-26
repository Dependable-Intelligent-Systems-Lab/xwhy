"""Distance metric types definitions."""

from __future__ import annotations

from enum import StrEnum


class DistanceType(StrEnum):
    """Enumeration for supported distance metrics."""

    COSINE = "cosine"
    WASSERSTEIN = "wasserstein"
    KS = "ks"
    CRAMER_VON_MISES = "cramer_von_mises"
    ANDERSON_DARLING = "anderson_darling"
    KUIPER = "kuiper"
    WMD = "wmd"
    DTS = "dts"

    @property
    def is_text_metric(self) -> bool:
        """Check if the metric is designed for text data."""
        return self == DistanceType.WMD

    @property
    def is_numeric_metric(self) -> bool:
        """Check if the metric is designed for numerical/image data."""
        return self != DistanceType.WMD

    @classmethod
    def from_str(cls, value: str | DistanceType) -> DistanceType:
        """Safely convert a string or enum instance to DistanceType."""
        try:
            return cls(value)
        except ValueError as err:
            valid_options = [item.value for item in cls]
            raise ValueError(
                f"'{value}' is not a valid DistanceType. "
                f"Please choose from: {valid_options}"
            ) from err
