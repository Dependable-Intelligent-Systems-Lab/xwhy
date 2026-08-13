"""Visualization functions for regression and fidelity metrics."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import auc, roc_curve

from xwhy.core.result import BaseXWhyResult
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


def plot_stability_visualization(
    result_one: BaseXWhyResult,
    result_two: BaseXWhyResult,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Visualize the stability flow between two explanation results.

    Plot Result 1 (top) and Result 2 (bottom) as heatmaps, with a 'Generative
    Model' node in the center. Edges are drawn from Result 1 words to the
    model, and from the model to Result 2 words.

    Args:
        result_one: The first explanation result object.
        result_two: The second explanation result object.
        **kwargs: Additional arguments including 'width' (float, default: 12.0),
            'height' (float, default: 8.0), and 'save_path' (str | None).

    """
    width: float = float(kwargs.get("width", 12.0))
    height: float = float(kwargs.get("height", 8.0))
    save_path: str | None = kwargs.get("save_path")

    words_one = result_one.feature_names
    words_two = result_two.feature_names
    coeffs_one = np.asarray(result_one.coefficients).flatten()
    coeffs_two = np.asarray(result_two.coefficients).flatten()

    # Normalize scores for coloring (global normalization across both prompts)
    all_scores = np.concatenate([coeffs_one, coeffs_two])
    denom = np.max(np.abs(all_scores))
    if denom == 0:
        denom = 1e-8

    # Helper to get color
    cmap = plt.cm.ScalarMappable(cmap=plt.cm.bwr)
    cmap.set_clim(0, 1)

    def get_color(score: float) -> str:
        norm_score = 0.5 * score / denom + 0.5
        r, g, b, _ = cmap.to_rgba(norm_score, bytes=True)
        return f"#{r:02x}{g:02x}{b:02x}"

    _, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Layout Configuration
    y_prompt_one = 85
    y_model = 50
    y_prompt_two = 15

    def calculate_x_positions(num_words: int) -> np.ndarray:
        return np.linspace(10, 90, num_words)

    x_pos_one = calculate_x_positions(len(words_one))  # type: ignore[arg-type]
    x_pos_two = calculate_x_positions(len(words_two))  # type: ignore[arg-type]
    x_model = 50  # Center

    # Draw Generative Model (Center)
    model_box = FancyBboxPatch(
        (x_model - 5, y_model - 3),
        10,
        6,
        boxstyle="round,pad=0.2",
        fc="#E0E0E0",
        ec="black",
        lw=2,
    )
    ax.add_patch(model_box)
    ax.text(
        x_model,
        y_model,
        "Generative\nModel",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    # Draw Prompt One (Top)
    for word, score, x in zip(words_one, coeffs_one, x_pos_one, strict=False):  # type: ignore[arg-type]
        color = get_color(score)

        ax.text(
            x,
            y_prompt_one,
            str(word),
            bbox={
                "facecolor": color,
                "pad": 5.0,
                "linewidth": 1,
                "boxstyle": "round,pad=0.5",
            },
            fontsize=12,
            ha="center",
        )

        ax.text(
            x,
            y_prompt_one - 5,
            f"{score:.2f}",
            fontsize=9,
            ha="center",
        )

        # Edge: Prompt One -> Model
        # Starting slightly below the score
        ax.annotate(
            "",
            xy=(x_model, y_model + 3),  # Target (Top of model box)
            xytext=(x, y_prompt_one - 6),  # Source (Bottom of score)
            arrowprops={
                "arrowstyle": "->",
                "color": "gray",
                "alpha": 0.5,
                "shrinkA": 5,
                "shrinkB": 5,
            },
        )

    # Draw Prompt Two (Bottom)
    for word, score, x in zip(words_two, coeffs_two, x_pos_two, strict=False):  # type: ignore[arg-type]
        color = get_color(score)

        # Word Box
        ax.text(
            x,
            y_prompt_two,
            str(word),
            bbox={
                "facecolor": color,
                "pad": 5.0,
                "linewidth": 1,
                "boxstyle": "round,pad=0.5",
            },
            fontsize=12,
            ha="center",
        )

        # Score Value
        ax.text(
            x,
            y_prompt_two - 5,
            f"{score:.2f}",
            fontsize=9,
            ha="center",
        )

        # Edge: Model -> Prompt Two
        # Target is top of the word box (approx y_prompt_two + padding)
        ax.annotate(
            "",
            xy=(x, y_prompt_two + 2),  # Target (Top of word box)
            xytext=(x_model, y_model - 3),  # Source (Bottom of model box)
            arrowprops={
                "arrowstyle": "->",
                "color": "gray",
                "alpha": 0.5,
                "shrinkA": 5,
                "shrinkB": 5,
            },
        )

    # Titles
    ax.text(50, 95, "Prompt 1 Source", fontsize=14, fontweight="bold", ha="center")
    ax.text(50, 5, "Prompt 2 Target", fontsize=14, fontweight="bold", ha="center")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.debug("Stability visualization saved to: %s", save_path)

    plt.show()


def plot_importance_roc_curve(
    result: BaseXWhyResult,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Plot the Receiver Operating Characteristic (ROC) curve for importance.

    Evaluate how well the model's contribution scores distinguish between
    ground-truth relevant and irrelevant tokens.

    Args:
        result: The explanation result object containing feature scores.
        **kwargs: Additional arguments including:
            - 'truth' (list[int]): Ground truth binary labels (Required).
            - 'title' (str): Title of the plot.
            - 'save_path' (str | None): Optional file path to save the plot.

    Raises:
        ValueError: If 'truth' is missing from kwargs or lengths mismatch.

    """
    truth = kwargs.get("truth")
    if truth is None:
        raise ValueError("The 'truth' parameter (list of ints) is required in kwargs.")

    title: str = str(kwargs.get("title", "ROC Curve - Token Importance"))
    save_path: str | None = kwargs.get("save_path")

    y_true = np.array(truth)
    y_scores = np.asarray(result.coefficients).flatten()

    if len(y_true) != len(y_scores):
        raise ValueError("Length of truth labels and result scores must match.")

    # Check if both classes are present
    if len(np.unique(y_true)) < 2:
        logger.warning(
            "ROC curve cannot be plotted with only one class in truth labels."
        )
        return

    # Calculate FPR, TPR and Area under the curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))

    plt.figure(figsize=(8, 6))

    # Plot the ROC curve
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (area = {roc_auc:.4f})",
    )

    # Plot the diagonal baseline (random classifier)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    plt.xlim((0.0, 1.0))
    plt.ylim((0.0, 1.05))
    plt.xlabel("False Positive Rate (Irrelevant tokens marked as important)")
    plt.ylabel("True Positive Rate (Relevant tokens correctly identified)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.debug("ROC curve saved to: %s", save_path)

    plt.show()
