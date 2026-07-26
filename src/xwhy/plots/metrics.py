"""Visualization functions for regression and fidelity metrics."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from xwhy.logger import logger
from xwhy.metrics.regression import RegressionMetricResult


def plot_fidelity(
    metrics: RegressionMetricResult,
    y_target: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    save_path: str | Path | None = None,
    show: bool = True,
) -> str | None:
    """Create and displays/saves an Actual vs Predicted fidelity scatter plot.

    Point size is determined by sample weight (larger means a more important sample).

    Args:
        metrics: Dataclass containing computed regression metrics.
        y_target: Array of actual target values (similarities).
        y_pred: Array of predicted values from the surrogate model.
        weights: Array of sample weights for points sizing.
        save_path: Optional path (including filename) to save the plot.
                   If None, the plot is not saved to disk.
        show: Whether to display the plot interactively.

    Returns:
        str | None: The absolute string path to the saved plot if save_path
                    was provided, otherwise None.

    Raises:
        ValueError: If input arrays have mismatched lengths.

    """
    if not (len(y_target) == len(y_pred) == len(weights)):
        raise ValueError("y_target, y_pred, and weights must have the same length.")

    # Normalize weights for point sizes (50 to 500)
    max_weight = weights.max()
    if max_weight > 0:
        point_sizes = (weights / max_weight) * 450 + 50
    else:
        point_sizes = np.full_like(weights, 100.0)

    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        y_target,
        y_pred,
        s=point_sizes,
        c=weights,
        cmap="plasma",
        alpha=0.75,
        edgecolors="black",
        linewidth=0.5,
    )

    # Perfect prediction line
    min_val = float(min(y_target.min(), y_pred.min()) * 0.98)
    max_val = float(max(y_target.max(), y_pred.max()) * 1.02)
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        lw=2.5,
        label="Perfect Prediction",
    )

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.set_xlabel("Actual Values", fontsize=13, fontweight="bold")
    ax.set_ylabel("Predicted Values", fontsize=13, fontweight="bold")
    ax.set_title(
        (
            "Actual vs Predicted Values\n"
            f"Weighted R² = {metrics.weighted_r2:.4f}  •  "
            f"Adjusted R² = {metrics.weighted_adj_r2:.4f}"
        ),
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Sample Weight (Importance)", rotation=270, labelpad=20, fontsize=11)

    resolved_path: str | None = None
    if save_path:
        path_obj = Path(save_path).resolve()
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_obj, dpi=300, bbox_inches="tight", facecolor="white")
        resolved_path = str(path_obj)
        logger.info("Fidelity plot saved: %s", resolved_path)

    if show:
        plt.show()

    plt.close(fig)

    return resolved_path
