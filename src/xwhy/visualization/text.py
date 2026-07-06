"""Native matplotlib text visualization implementations."""

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

from xwhy.visualization.base import BaseTextVisualizer


class NativeHeatmapVisualizer(BaseTextVisualizer):
    """Native matplotlib implementation of text heatmap visualization."""

    def plot(
        self,
        words: Sequence[str],
        scores: np.ndarray,
        title: str = "",
        width: float = 10.0,
        height: float = 0.5,
        verbose: int = 0,
        max_word_per_line: int = 20,
        word_spacing: int = 20,
        score_fontsize: int = 10,
        save_path: str | None = None,
        **kwargs: object,
    ) -> None:
        """Plot a heatmap-like visualization over text tokens.

        Each token is shown inside a colored box based on its score, with the
        numeric score displayed underneath it.

        Args:
            words: Sequence of text tokens.
            scores: Array of per-token scores.
            title: Title shown on the plot.
            width: Figure width in inches.
            height: Figure height in inches.
            verbose: If 0, hide axes (clean output).
            max_word_per_line: Max number of tokens per visual line.
            word_spacing: Horizontal spacing between tokens.
            score_fontsize: Font size for numeric score labels.
            save_path: Optional save path.
            **kwargs: Additional ignored arguments for interface compatibility.

        """
        num_lines = math.ceil(len(words) / max_word_per_line)
        dynamic_height = max(2.0, num_lines * height * 2.5)

        _ = plt.figure(figsize=(width, dynamic_height))
        ax = plt.gca()

        ax.set_title(title, loc="left", pad=10)

        # Color map normalization
        cmap = plt.cm.ScalarMappable(cmap=plt.cm.bwr)
        cmap.set_clim(0, 1)

        denom = np.max(np.abs(scores))
        if denom == 0:
            denom = 1e-8  # avoid division by zero
        normalized = 0.5 * scores / denom + 0.5

        canvas = ax.figure.canvas
        transform = ax.transData

        y = 0.0

        for i, (word, score, ns) in enumerate(
            zip(words, scores, normalized, strict=False)
        ):
            r, g, b, _ = cmap.to_rgba(ns, bytes=True)
            color = f"#{r:02x}{g:02x}{b:02x}"

            # draw token
            txt = ax.text(
                0.0,
                y,
                word,
                bbox={
                    "facecolor": color,
                    "pad": 5.0,
                    "linewidth": 1,
                    "boxstyle": "round,pad=0.5",
                },
                transform=transform,
                fontsize=14,
            )
            txt.draw(canvas.get_renderer())  # type: ignore
            ex = txt.get_window_extent()

            # draw numeric score under token
            score_txt = ax.text(
                0.01,
                y - 0.5,
                f"{score:.2f}",
                transform=transform,
                fontsize=score_fontsize,
                ha="center",
            )
            score_txt.draw(canvas.get_renderer())  # type: ignore

            # new transform for next token
            if (i + 1) % max_word_per_line == 0:
                y -= 2.5
                transform = ax.transData
            else:
                transform = transforms.offset_copy(
                    txt._transform,  # type: ignore
                    x=ex.width + word_spacing,
                    units="dots",
                )

        ax.set_ylim(y - 1.5, 0.5)
        ax.set_xlim(0, 1)

        if verbose == 0:
            ax.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
