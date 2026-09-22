"""Native matplotlib text plot implementations."""

import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

from xwhy.plots.base import BaseTextPlotter


class NativeHeatmapPlotter(BaseTextPlotter):
    """Native matplotlib implementation of text heatmap plot."""

    def plot(
        self,
        words: Sequence[str],
        scores: np.ndarray,
        title: str = "",
        width: float = 10.0,
        height: float = 0.5,
        verbose: int = 0,
        max_word_per_line: int = 20,
        word_spacing: int = 30,
        score_fontsize: int = 10,
        line_spacing: float = 1.5,
        score_gap: float = 0.7,
        horizontal_margin: float = 0.05,
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
            line_spacing: Vertical spacing multiplier between lines.
            score_gap: Vertical gap beneath the token for the score label.
            horizontal_margin: Left and right figure padding (fraction of width).
            save_path: Optional save path.
            **kwargs: Additional ignored arguments for interface compatibility.

        """
        num_lines = max(1, math.ceil(len(words) / max_word_per_line))
        # Keep the original density formula, but drop the large forced minimum
        # so short lists do not get an artificially tall empty figure.
        dynamic_height = max(1.15, num_lines * height * (line_spacing + 0.2))

        _ = plt.figure(figsize=(width, dynamic_height))
        ax = plt.gca()

        # Formula-based title spacing and font size based on the number of lines.
        title_pad = 10 + 8 * math.sqrt(num_lines)
        title_fontsize = 14 + 2 * math.sqrt(num_lines)

        ax.set_title(
            title,
            loc="left",
            pad=title_pad,
            fontsize=title_fontsize,
        )

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

            # draw numeric score centered under token
            score_transform = transforms.offset_copy(
                transform,
                x=ex.width / 2,
                units="dots",
            )
            score_txt = ax.text(
                0.0,
                y - score_gap,
                f"{score:.2f}",
                transform=score_transform,
                fontsize=score_fontsize,
                ha="center",
            )
            score_txt.draw(canvas.get_renderer())  # type: ignore

            # new transform for next token
            if (i + 1) % max_word_per_line == 0:
                y -= line_spacing
                transform = ax.transData
            else:
                transform = transforms.offset_copy(
                    txt._transform,  # type: ignore
                    x=ex.width + word_spacing,
                    units="dots",
                )

        # Tight top so the first token sits close to the title area.
        # The visual title→box gap is then controlled by subplots_adjust(top=…)
        # and stays consistent for short and long lists.
        data_top = 0.15
        ax.set_ylim(y - (score_gap + 0.6), data_top)
        ax.set_xlim(0, 1)

        if verbose == 0:
            ax.axis("off")

        # Convert margin to a safe fraction of figure width
        margin_frac = float(horizontal_margin)
        if margin_frac >= 0.5:
            # Treat as pixels (assuming ~100 dpi)
            margin_frac = margin_frac / (width * 100.0) if margin_frac >= 1.0 else 0.49

        margin_frac = max(0.0, min(0.49, margin_frac))

        # Layout with full-width axes so the title (loc="left") and the tokens
        # share the exact same left edge.  top/bottom leave room for the title
        # and the score labels.  Horizontal margins are applied afterwards.
        plt.subplots_adjust(
            left=0.0,
            right=1.0,
            top=0.82,
            bottom=0.08,
        )

        # Common: compute the padded bbox (horizontal margin + small top air)
        fig = plt.gcf()
        fig.canvas.draw()  # ensure final positions are known
        tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())  # type: ignore[attr-defined]
        h_pad = margin_frac * width
        top_pad = 0.18  # inches of air above the title
        expanded_bbox = transforms.Bbox.from_extents(
            tight_bbox.x0 - h_pad,
            tight_bbox.y0,
            tight_bbox.x1 + h_pad,
            tight_bbox.y1 + top_pad,
        )

        if save_path:
            fig.savefig(save_path, bbox_inches=expanded_bbox)
            plt.close()
        else:
            # Jupyter's inline backend (and many notebook front-ends) apply
            # bbox_inches="tight" when rendering, which crops the side margins
            # created by subplots_adjust.  To guarantee the margins are visible
            # we render the same padded image that save_path would produce and
            # display it via IPython when available.
            try:
                import io

                from IPython.display import Image, display

                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches=expanded_bbox)
                buf.seek(0)
                display(Image(data=buf.getvalue()))  # type: ignore[no-untyped-call]
                plt.close()
            except Exception:
                # Non-notebook / plain Python: fall back to a normal show with
                # the axes inset so the window still shows side margins.
                plt.subplots_adjust(
                    left=margin_frac,
                    right=1.0 - margin_frac,
                    top=0.82,
                    bottom=0.08,
                )
                plt.show()
