"""Image overlays of pixel-level attributions."""

from __future__ import annotations

import json
import random
import string
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

from .base import (
    RED_TRANSPARENT_BLUE,
    _as_explanation,
    _finish_matplotlib,
)

# ---------------------------------------------------------------------------
# Data summary helpers
# ---------------------------------------------------------------------------


class Data:
    """Marker base class for summarized background datasets."""


class DenseData(Data):
    """Dense weighted summary of a feature matrix.

    Parameters
    ----------
    data :
        Summary matrix (``n_samples x n_features`` or transposed).
    group_names :
        Human-readable names for each feature group.
    *args :
        Optional ``groups`` (list of index arrays) and ``weights``
        (per-sample weights).

    """

    def __init__(
        self,
        data: NDArray[Any],
        group_names: Sequence[str],
        *args: Any,  # noqa: ANN401
    ) -> None:
        """Build a dense summary, validating group and weight lengths."""
        if len(args) > 0 and args[0] is not None:
            self.groups: list[NDArray[Any]] = list(args[0])
        else:
            self.groups = [np.array([i]) for i in range(len(group_names))]

        total_group_cols = sum(len(g) for g in self.groups)
        num_samples = data.shape[0]
        transposed = False
        if total_group_cols != data.shape[1]:
            transposed = True
            num_samples = data.shape[1]

        valid_shape = (not transposed and total_group_cols == data.shape[1]) or (
            transposed and total_group_cols == data.shape[0]
        )
        if not valid_shape:
            msg = "# of names must match data matrix!"
            raise ValueError(msg)

        weights = args[1] if len(args) > 1 else np.ones(num_samples)
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.sum(weights)
        weight_len = len(weights)
        valid_weights = (not transposed and weight_len == data.shape[0]) or (
            transposed and weight_len == data.shape[1]
        )
        if not valid_weights:
            msg = "# of weights must match data matrix!"
            raise ValueError(msg)

        self.transposed = transposed
        self.group_names = list(group_names)
        self.data = data
        self.weights = weights
        self.groups_size = len(self.groups)


def kmeans(
    x: NDArray[Any] | pd.DataFrame,
    k: int,
    *,
    round_values: bool = True,
) -> DenseData:
    """Summarize a dataset with *k* mean samples weighted by cluster size.

    Parameters
    ----------
    x :
        Matrix of data samples (``# samples x # features``). May be a
        NumPy array, pandas DataFrame, or any scipy sparse matrix.
    k :
        Number of means to use for the approximation.
    round_values :
        When ``True``, round each mean coordinate to the nearest observed
        value in that column so discrete features stay valid.

    Returns
    -------
    DenseData
        Weighted cluster centers.

    """
    if isinstance(x, pd.DataFrame):
        group_names: Sequence[str] = list(x.columns.astype(str))
        matrix: NDArray[Any] = x.values
    else:
        matrix = np.asarray(x) if not scipy.sparse.issparse(x) else x
        group_names = [str(i) for i in range(matrix.shape[1])]

    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    matrix = imp.fit_transform(matrix)

    # n_init fixed for consistent behaviour across sklearn versions
    model = KMeans(n_clusters=k, random_state=0, n_init=10).fit(matrix)

    if round_values:
        for i in range(k):
            for j in range(matrix.shape[1]):
                if scipy.sparse.issparse(matrix):
                    col = matrix[:, j].toarray().flatten()  # type: ignore[attr-defined]
                else:
                    col = matrix[:, j]
                ind = int(np.argmin(np.abs(col - model.cluster_centers_[i, j])))
                model.cluster_centers_[i, j] = matrix[ind, j]

    return DenseData(
        model.cluster_centers_,
        group_names,
        None,
        1.0 * np.bincount(model.labels_),
    )


# ---------------------------------------------------------------------------
# Image display helpers
# ---------------------------------------------------------------------------


def _to_display_image(
    x_curr: NDArray[Any],
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Convert a single image to grayscale and RGB-display forms.

    Returns
    -------
    gray, display
        Grayscale array used under the SHAP overlay, and the array shown
        in the original-image column.

    """
    # Single-channel 3-D -> 2-D
    if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
        x_curr = x_curr.reshape(x_curr.shape[:2])

    if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
        gray = (
            0.2989 * x_curr[:, :, 0]
            + 0.5870 * x_curr[:, :, 1]
            + 0.1140 * x_curr[:, :, 2]
        )
        return gray, x_curr

    if len(x_curr.shape) == 3:
        gray = x_curr.mean(2)
        # Non-RGB multi-channel: project onto 3 k-means centres as RGB
        flat_vals = x_curr.reshape(
            [x_curr.shape[0] * x_curr.shape[1], x_curr.shape[2]]
        ).T
        flat_vals = (flat_vals.T - flat_vals.mean(1)).T
        means = kmeans(flat_vals, 3, round_values=False).data.T.reshape(
            [x_curr.shape[0], x_curr.shape[1], 3]
        )
        lo = np.percentile(means, 0.5, (0, 1))
        hi = np.percentile(means, 99.5, (0, 1))
        display = (means - lo) / (hi - np.percentile(means, 1, (0, 1)) + 1e-12)
        display = np.clip(display, 0, 1)
        return gray, display

    return x_curr, x_curr


def _shap_image(
    shap_values: Any,  # noqa: ANN401
    pixel_values: NDArray[Any] | None = None,
    labels: Sequence[str] | NDArray[Any] | None = None,
    true_labels: Sequence[str] | None = None,
    width: float = 20,
    aspect: float = 0.2,
    hspace: float | str = 0.2,
    labelpad: float | None = None,
    cmap: Any = None,  # noqa: ANN401
    vmax: float | None = None,
    show: bool = True,
) -> Figure:
    r"""Plot SHAP values overlaid on image inputs.

    Parameters
    ----------
    shap_values :
        List of arrays ``(# samples x H x W x C)``, one per model output,
        or a single array / Explanation object.
    pixel_values :
        Pixel matrix matching each SHAP array. Required unless *shap_values*
        is an Explanation carrying ``.data``.
    labels :
        Optional ``(# samples x n_outputs)`` names for model outputs.
    true_labels :
        Optional true labels shown above the original-image column.
    width :
        Maximum figure width in inches.
    aspect :
        Colorbar aspect ratio relative to figure width.
    hspace :
        Subplot vertical spacing, or ``\"auto\"`` for ``tight_layout``.
    labelpad :
        Padding around title labels.
    cmap :
        Colormap for SHAP overlays (default ``RED_TRANSPARENT_BLUE``).
    vmax :
        Symmetric color scale limit; defaults to the 99.9th percentile of
        absolute SHAP values.
    show :
        Accepted for API compatibility; display is handled by the caller.

    Returns
    -------
    Figure
        The constructed matplotlib figure.

    """
    _ = show
    if cmap is None:
        cmap = RED_TRANSPARENT_BLUE

    # Support passing an Explanation-like object
    if type(shap_values).__name__ == "Explanation":
        shap_exp = shap_values
        if shap_exp.values.ndim >= 5:
            shap_list: list[NDArray[Any]] = [
                shap_exp.values[..., i] for i in range(shap_exp.values.shape[-1])
            ]
        else:
            shap_list = [shap_exp.values]
        if pixel_values is None:
            pixel_values = shap_exp.data
        if labels is None:
            labels = shap_exp.output_names
    else:
        if pixel_values is None:
            msg = (
                "The input pixel_values must be a numpy array or an "
                "Explanation object must be provided!"
            )
            raise AssertionError(msg)
        if not isinstance(shap_values, list):
            shap_list = [cast("NDArray[Any]", shap_values)]
        else:
            shap_list = list(shap_values)

    if pixel_values is None:
        msg = "pixel_values is required"
        raise ValueError(msg)

    if len(shap_list[0].shape) == 3:
        shap_list = [v.reshape(1, *v.shape) for v in shap_list]
        pixel_values = pixel_values.reshape(1, *pixel_values.shape)

    labels_arr: NDArray[Any] | None = None
    if labels is not None:
        labels_arr = np.asarray(labels)
        labels_arr = labels_arr.reshape(-1, len(shap_list))

    label_kwargs: dict[str, float] = {} if labelpad is None else {"pad": labelpad}

    images = pixel_values
    fig_size = np.array(
        [3 * (len(shap_list) + 1), 2.5 * (images.shape[0] + 1)],
        dtype=float,
    )
    if fig_size[0] > width:
        fig_size *= width / fig_size[0]

    fig, axes = plt.subplots(
        nrows=images.shape[0],
        ncols=len(shap_list) + 1,
        figsize=fig_size,
        squeeze=False,
    )

    im = None
    for row in range(images.shape[0]):
        x_curr = images[row].copy()
        gray, display = _to_display_image(x_curr)

        axes[row, 0].imshow(display, cmap=plt.get_cmap("gray"))
        if true_labels:
            axes[row, 0].set_title(true_labels[row], **label_kwargs)
        axes[row, 0].axis("off")

        if len(shap_list[0][row].shape) == 2:
            abs_vals = np.stack(
                [np.abs(shap_list[i]) for i in range(len(shap_list))],
                0,
            ).flatten()
        else:
            abs_vals = np.stack(
                [np.abs(shap_list[i].sum(-1)) for i in range(len(shap_list))],
                0,
            ).flatten()

        max_val = float(np.nanpercentile(abs_vals, 99.9)) if vmax is None else vmax

        for i in range(len(shap_list)):
            if labels_arr is not None and (labels_arr.shape[0] > 1 or row == 0):
                axes[row, i + 1].set_title(labels_arr[row, i], **label_kwargs)

            sv = (
                shap_list[i][row]
                if len(shap_list[i][row].shape) == 2
                else shap_list[i][row].sum(-1)
            )
            axes[row, i + 1].imshow(
                gray,
                cmap=plt.get_cmap("gray"),
                alpha=0.15,
                extent=(-1, sv.shape[1], sv.shape[0], -1),
            )
            im = axes[row, i + 1].imshow(sv, cmap=cmap, vmin=-max_val, vmax=max_val)
            axes[row, i + 1].axis("off")

    if hspace == "auto":
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=float(hspace))

    if im is not None:
        colorbar = fig.colorbar(
            im,
            ax=np.ravel(axes).tolist(),
            label="XWhy value",
            orientation="horizontal",
            aspect=fig_size[0] / aspect,
        )
        colorbar.locator = MaxNLocator(nbins=5)
        colorbar.update_ticks()
        colorbar.outline.set_visible(False)

    return fig


def ordinal_str(n: int) -> str:
    r"""Convert a number to an ordinal string (1st, 2nd, 3rd, …).

    Args:
        n: Non-negative integer.

    Returns:
        Ordinal form such as ``\"1st\"`` or ``\"22nd\"``.

    """
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(
        4 if 10 <= n % 100 < 20 else n % 10,
        "th",
    )
    return f"{n}{suffix}"


def _escape_token(token: object) -> str:
    """HTML-escape a model output token for safe embedding."""
    text = str(token)
    return (
        text.replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace(" ##", "")
        .replace("▁", "")
        .replace("Ġ", "")
    )


def _flatten_output_names(model_output: Any) -> list[Any]:  # noqa: ANN401
    """Normalise ``output_names`` to a flat 1-D sequence of tokens."""
    if hasattr(model_output, "flatten"):
        flat = model_output.flatten()
        return list(flat)
    if (
        isinstance(model_output, list)
        and len(model_output) == 1
        and isinstance(model_output[0], (list, np.ndarray))
    ):
        return list(model_output[0])
    if isinstance(model_output, np.ndarray) and model_output.ndim > 1:
        return list(model_output.flatten())
    return list(model_output)


def _build_output_tokens_html(uuid: str, model_output: list[Any]) -> str:
    """Build clickable HTML spans for each output token."""
    parts: list[str] = []
    for i, token in enumerate(model_output):
        escaped = _escape_token(token)
        parts.append(
            "<div style='display:inline; text-align:center;'>"
            f"<div id='{uuid}_output_flat_value_label_{i}' "
            "style='display:none;color: #999; padding-top: 0px; "
            "font-size:12px;'></div>"
            f"<div id='{uuid}_output_flat_token_{i}' "
            "style='display: inline; background:transparent; "
            "border-radius: 3px; padding: 0px;cursor: default;"
            "cursor: pointer;' "
            f'onmouseover="onMouseHoverFlat_{uuid}(this.id)" '
            f'onmouseout="onMouseOutFlat_{uuid}(this.id)" '
            f'onclick="onMouseClickFlat_{uuid}(this.id)">'
            f"{escaped} </div></div>"
        )
    return "".join(parts)


def _grayscale_rgba(image_data: NDArray[Any]) -> NDArray[Any]:
    """Build a 4-channel grayscale RGBA image from RGB(A) input."""
    image_height, image_width = image_data.shape[0], image_data.shape[1]
    gray = np.ones((image_height, image_width, 4)) * 255 * 0.5
    mean_ch = np.mean(image_data, axis=2).astype(int)
    gray[:, :, 0] = mean_ch
    gray[:, :, 1] = mean_ch
    gray[:, :, 2] = mean_ch
    return gray


def _shap_color_maps(
    uuid: str,
    shap_values: Any,  # noqa: ANN401
    model_output: list[Any],
) -> dict[str, Any]:
    """Map each output-token DOM id to an RGBA colour grid."""
    shap_maps = shap_values.values[:, :, 0, :]
    max_val = float(np.nanpercentile(np.abs(shap_values.values), 99.9))
    color_dict: dict[str, Any] = {}
    for index in range(len(model_output)):
        key = f"{uuid}_output_flat_token_{index}"
        scaled = 0.5 + 0.5 * shap_maps[:, :, index] / max_val
        rgba = (RED_TRANSPARENT_BLUE(scaled) * 255).astype(int)
        color_dict[key] = rgba.tolist()
    return color_dict


def _image_viz_html(
    uuid: str,
    output_text_html: str,
) -> str:
    """Return the static HTML shell for the image/token viewer."""
    return f"""
        <div id="{uuid}_image_viz" class="{uuid}_image_viz_content">
          <div id="{uuid}_image_viz_header"
               style="padding:15px;margin:5px;font-family:sans-serif;
                      font-weight:bold;">
            <div style="display:inline">
              <span style="font-size: 20px;"> Input/Output - Heatmap </span>
            </div>
          </div>
          <div id="{uuid}_image_viz_content" style="display:flex;">
            <div id="{uuid}_image_viz_input_container"
                 style="padding:15px;border-style:solid;margin:5px;flex:2;">
              <div id="{uuid}_image_viz_input_header"
                   style="margin:5px;font-weight:bold;font-family:sans-serif;
                          margin-bottom:10px">
                Input Image
              </div>
              <div id="{uuid}_image_viz_input_content"
                   style="margin:5px;font-family:sans-serif;">
                  <canvas id="{uuid}_image_canvas"
                          style="cursor:grab;width:100%;max-height:500px;">
                  </canvas>
                  <br><br>
                  <div id="{uuid}_tools">
                      <div id="{uuid}_zoom">
                        <span style="font-size:12px;margin-right:15px;">
                          Zoom
                        </span>
                        <button id="{uuid}_minus_button" class="zoom-button"
                                onclick="{uuid}_zoom(-1)"
                                style="background-color: #555555;color: white;
                                       border:none;font-size:15px;">-</button>
                        <button id="{uuid}_plus_button" class="zoom-button"
                                onclick="{uuid}_zoom(1)"
                                style="background-color: #555555;color: white;
                                       border:none;font-size:15px;">+</button>
                        <button id="{uuid}_reset_button" class="zoom-button"
                                onclick="{uuid}_reset()"
                                style="background-color: #555555;color: white;
                                       border:none;font-size:15px;">
                          Reset
                        </button>
                      </div>
                      <br>
                      <div id="{uuid}_opacity" style="display:none">
                      <span style="font-size:12px;margin-right:15px;">
                        XWhy-Overlay Opacity
                      </span>
                      <input type="range" min="1" max="100" value="35"
                             style="width:100px"
                             oninput="{uuid}_set_opacity(this.value)">
                      </div>
                  </div>
              </div>
            </div>
            <div id="{uuid}_image_viz_output_container"
                 style="padding:15px;border-style:solid;margin:5px;flex:1;">
              <div id="{uuid}_image_viz_output_header"
                   style="margin:5px;font-weight:bold;font-family:sans-serif;
                          margin-bottom:10px">
                Output Text
              </div>
              <div id="{uuid}_image_viz_output_content"
                   style="margin:5px;font-family:sans-serif;">
                  {output_text_html}
              </div>
            </div>
          </div>
        </div>
    """


def _image_viz_script(
    uuid: str,
    image_data_json: str,
    image_data_gray_scale_json: str,
    shap_values_color_dict_json: str,
    image_height: int,
    image_width: int,
) -> str:
    """Return the JavaScript that drives canvas zoom and SHAP overlay."""
    return f"""
        <script>
            var {uuid}_heatmap_flat_state = null;
            var {uuid}_opacity = 0.35

            function onMouseHoverFlat_{uuid}(id) {{
                if ({uuid}_heatmap_flat_state === null) {{
                    document.getElementById(id).style.backgroundColor =
                        "grey";
                    {uuid}_update_image_and_overlay(id);
                }}
            }}

            function onMouseOutFlat_{uuid}(id) {{
                if ({uuid}_heatmap_flat_state === null) {{
                    document.getElementById(id).style.backgroundColor =
                        "transparent";
                    {uuid}_update_image_and_overlay(null);
                }}
            }}

            function onMouseClickFlat_{uuid}(id) {{
                if ({uuid}_heatmap_flat_state === null) {{
                    document.getElementById(id).style.backgroundColor =
                        "grey";
                    document.getElementById('{uuid}_opacity').style.display =
                        "block";
                    {uuid}_update_image_and_overlay(id);
                    {uuid}_heatmap_flat_state = id;
                }}
                else {{
                    if ({uuid}_heatmap_flat_state === id) {{
                        document.getElementById(id).style.backgroundColor =
                            "transparent";
                        document.getElementById(
                            '{uuid}_opacity').style.display = "none";
                        {uuid}_update_image_and_overlay(null);
                        {uuid}_heatmap_flat_state = null;
                    }}
                    else {{
                        document.getElementById(
                            {uuid}_heatmap_flat_state
                        ).style.backgroundColor = "transparent";
                        document.getElementById(id).style.backgroundColor =
                            "grey";
                        {uuid}_update_image_and_overlay(id)
                        {uuid}_heatmap_flat_state = id
                    }}
                }}
            }}

            const {uuid}_image_data_matrix = {image_data_json};
            const {uuid}_image_data_gray_scale =
                {image_data_gray_scale_json};
            const {uuid}_image_height = {image_height};
            const {uuid}_image_width = {image_width};
            const {uuid}_shap_values_color_dict =
                {shap_values_color_dict_json};

            {uuid}_canvas = document.getElementById(
                '{uuid}_image_canvas');
            {uuid}_context = {uuid}_canvas.getContext('2d');

            var {uuid}_imageData = {uuid}_convert_image_matrix_to_data(
                {uuid}_image_data_matrix, {image_height}, {image_width},
                {uuid}_context);
            var {uuid}_currImagData = {uuid}_imageData;

            {uuid}_trackTransforms({uuid}_context);
            initial_scale_factor = Math.min(
                {uuid}_canvas.height/{uuid}_image_height,
                {uuid}_canvas.width/{uuid}_image_width);
            {uuid}_context.scale(
                initial_scale_factor, initial_scale_factor);

            function {uuid}_update_image_and_overlay(selected_id) {{
                if (selected_id == null) {{
                    {uuid}_currImagData = {uuid}_imageData;
                    {uuid}_redraw();
                }}
                else {{
                    {uuid}_currImagData = {uuid}_blend_image_shap_map(
                        {uuid}_image_data_gray_scale,
                        {uuid}_shap_values_color_dict[selected_id],
                        {image_height}, {image_width},
                        {uuid}_opacity, {uuid}_context);
                    {uuid}_redraw();
                }}
            }}

            function {uuid}_set_opacity(value) {{
                {uuid}_opacity = value/100;
                if ({uuid}_heatmap_flat_state !== null ) {{
                    {uuid}_currImagData = {uuid}_blend_image_shap_map(
                        {uuid}_image_data_gray_scale,
                        {uuid}_shap_values_color_dict[
                            {uuid}_heatmap_flat_state],
                        {image_height}, {image_width},
                        {uuid}_opacity, {uuid}_context);
                    {uuid}_redraw();
                }}
            }}

            function {uuid}_redraw() {{
                var p1 = {uuid}_context.transformedPoint(0, 0);
                var p2 = {uuid}_context.transformedPoint(
                    {uuid}_canvas.width, {uuid}_canvas.height);
                {uuid}_context.clearRect(
                    p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
                {uuid}_context.save();
                {uuid}_context.setTransform(1, 0, 0, 1, 0, 0);
                {uuid}_context.clearRect(
                    0, 0, {uuid}_canvas.width, {uuid}_canvas.height);
                {uuid}_context.restore();
                createImageBitmap(
                    {uuid}_currImagData,
                    {{ premultiplyAlpha: 'premultiply' }}
                ).then(function(imgBitmap) {{
                    {uuid}_context.drawImage(imgBitmap, 0, 0);
                }});
            }}
            {uuid}_redraw();
            {uuid}_context.save();

            var lastX = {uuid}_canvas.width / 2,
                lastY = {uuid}_canvas.height / 2;
            var dragStart, dragged;

            {uuid}_canvas.addEventListener('mousedown', function(evt) {{
                document.body.style.mozUserSelect =
                    document.body.style.webkitUserSelect =
                    document.body.style.userSelect = 'none';
                lastX = evt.offsetX ||
                    (evt.pageX - {uuid}_canvas.offsetLeft);
                lastY = evt.offsetY ||
                    (evt.pageY - {uuid}_canvas.offsetTop);
                dragStart = {uuid}_context.transformedPoint(
                    lastX, lastY);
                dragged = false;
                document.getElementById(
                    '{uuid}_image_canvas').style.cursor = 'grabbing';
            }}, false);

            {uuid}_canvas.addEventListener('mousemove', function(evt) {{
                lastX = evt.offsetX ||
                    (evt.pageX - {uuid}_canvas.offsetLeft);
                lastY = evt.offsetY ||
                    (evt.pageY - {uuid}_canvas.offsetTop);
                dragged = true;
                if (dragStart) {{
                    var pt = {uuid}_context.transformedPoint(
                        lastX, lastY);
                    {uuid}_context.translate(
                        pt.x - dragStart.x, pt.y - dragStart.y);
                    {uuid}_redraw();
                }}
            }}, false);

            {uuid}_canvas.addEventListener('mouseup', function(evt) {{
                dragStart = null;
                document.getElementById(
                    '{uuid}_image_canvas').style.cursor = 'grab';
            }}, false);

            var scaleFactor = 1.1;

            var {uuid}_zoom = function(clicks) {{
                var pt = {uuid}_context.transformedPoint(lastX, lastY);
                {uuid}_context.translate(pt.x, pt.y);
                var factor = Math.pow(scaleFactor, clicks);
                {uuid}_context.scale(factor, factor);
                {uuid}_context.translate(-pt.x, -pt.y);
                {uuid}_redraw();
            }}

            var {uuid}_reset = function(clicks) {{
                {uuid}_context.restore();
                {uuid}_redraw();
                {uuid}_context.save();
            }}

            var handleScroll = function(evt) {{
                var delta = evt.wheelDelta ? evt.wheelDelta / 40
                    : evt.detail ? -evt.detail : 0;
                if (delta) {uuid}_zoom(delta);
                return evt.preventDefault() && false;
            }}

            {uuid}_canvas.addEventListener(
                'DOMMouseScroll', handleScroll, false);
            {uuid}_canvas.addEventListener(
                'mousewheel', handleScroll, false);

            function {uuid}_trackTransforms(ctx) {{
                var svg = document.createElementNS(
                    "http://www.w3.org/2000/svg", 'svg');
                var xform = svg.createSVGMatrix();
                ctx.getTransform = function() {{
                    return xform;
                }}

                var savedTransforms = [];
                var save = ctx.save;
                ctx.save = function() {{
                    savedTransforms.push(xform.translate(0, 0));
                    return save.call(ctx);
                }}

                var restore = ctx.restore;
                ctx.restore = function() {{
                    xform = savedTransforms.pop();
                    return restore.call(ctx);
                }}

                var scale = ctx.scale;
                ctx.scale = function(sx, sy) {{
                    xform = xform.scaleNonUniform(sx, sy);
                    return scale.call(ctx, sx, sy);
                }}

                var rotate = ctx.rotate;
                ctx.rotate = function(radians) {{
                    xform = xform.rotate(radians * 180 / Math.PI);
                    return rotate.call(ctx, radians);
                }}

                var translate = ctx.translate;
                ctx.translate = function(dx, dy) {{
                    xform = xform.translate(dx, dy);
                    return translate.call(ctx, dx, dy);
                }}

                var transform = ctx.transform;
                ctx.transform = function(a, b, c, d, e, f) {{
                    var m2 = svg.createSVGMatrix();
                    m2.a = a; m2.b = b; m2.c = c;
                    m2.d = d; m2.e = e; m2.f = f;
                    xform = xform.multiply(m2);
                    return transform.call(ctx, a, b, c, d, e, f);
                }}

                var setTransform = ctx.setTransform;
                ctx.setTransform = function(a, b, c, d, e, f) {{
                    xform.a = a; xform.b = b; xform.c = c;
                    xform.d = d; xform.e = e; xform.f = f;
                    return setTransform.call(ctx, a, b, c, d, e, f);
                }}

                var pt = svg.createSVGPoint();
                ctx.transformedPoint = function(x, y) {{
                    pt.x = x;
                    pt.y = y;
                    return pt.matrixTransform(xform.inverse());
                }}
            }}

            function {uuid}_convert_image_matrix_to_data(
                    image_data_matrix, image_height, image_width,
                    context) {{
                var imageData = context.createImageData(
                    image_height, image_width);
                for (var row_index = 0; row_index < image_height;
                     row_index++) {{
                    for (var col_index = 0; col_index < image_width;
                         col_index++) {{
                        index = (row_index * image_width + col_index)
                            * 4;
                        imageData.data[index + 0] =
                            image_data_matrix[row_index][col_index][0];
                        imageData.data[index + 1] =
                            image_data_matrix[row_index][col_index][1];
                        imageData.data[index + 2] =
                            image_data_matrix[row_index][col_index][2];
                        imageData.data[index + 3] = 255;
                    }}
                }}
                return imageData;
            }}

            function {uuid}_blend_image_shap_map(
                    image_data_matrix, shap_color_map, image_height,
                    image_width, alpha, context) {{
                var blendedImageData = context.createImageData(
                    image_height, image_width);
                for (var row_index = 0; row_index < image_height;
                     row_index++) {{
                    for (var col_index = 0; col_index < image_width;
                         col_index++) {{
                        index = (row_index * image_width + col_index)
                            * 4;
                        blendedImageData.data[index + 0] =
                            image_data_matrix[row_index][col_index][0]
                            * alpha
                            + (shap_color_map[row_index][col_index][0])
                            * (1 - alpha);
                        blendedImageData.data[index + 1] =
                            image_data_matrix[row_index][col_index][1]
                            * alpha
                            + (shap_color_map[row_index][col_index][1])
                            * (1 - alpha);
                        blendedImageData.data[index + 2] =
                            image_data_matrix[row_index][col_index][2]
                            * alpha
                            + (shap_color_map[row_index][col_index][2])
                            * (1 - alpha);
                        blendedImageData.data[index + 3] =
                            image_data_matrix[row_index][col_index][3]
                            * alpha
                            + (shap_color_map[row_index][col_index][3])
                            * (1 - alpha);
                    }}
                }}
                return blendedImageData;
            }}
        </script>
    """


def _shap_image_to_text(shap_values: Any) -> None:  # noqa: ANN401
    """Plot SHAP values for image inputs with text outputs (IPython).

    Args:
        shap_values: Explanation whose ``values`` have shape
            ``(width, height, channels, num_output_tokens)`` per sample,
            or a batch with a leading sample axis.

    """
    try:
        from IPython.display import HTML, display

        have_ipython = True
    except ImportError:
        have_ipython = False

    if not have_ipython:  # pragma: no cover
        msg = (
            "IPython is required for this function but is not installed. "
            "Fix this with `pip install ipython`."
        )
        raise ImportError(msg)

    if len(shap_values.values.shape) == 5:
        for i in range(shap_values.values.shape[0]):
            display(HTML(f"<br/><b>{ordinal_str(i)} instance:</b><br/>"))  # type: ignore[no-untyped-call]
            _shap_image_to_text(shap_values[i])
        return

    uuid = "".join(random.choices(string.ascii_lowercase, k=20))

    model_output = _flatten_output_names(shap_values.output_names)
    output_text_html = _build_output_tokens_html(uuid, model_output)

    image_data = shap_values.data
    image_height = int(image_data.shape[0])
    image_width = int(image_data.shape[1])
    image_data_gray_scale = _grayscale_rgba(image_data)
    shap_values_color_dict = _shap_color_maps(uuid, shap_values, model_output)

    image_data_json = json.dumps(shap_values.data.astype(int).tolist())
    shap_values_color_dict_json = json.dumps(shap_values_color_dict)
    image_data_gray_scale_json = json.dumps(image_data_gray_scale.astype(int).tolist())

    html = _image_viz_html(uuid, output_text_html)
    script = _image_viz_script(
        uuid,
        image_data_json,
        image_data_gray_scale_json,
        shap_values_color_dict_json,
        image_height,
        image_width,
    )
    display(HTML(html + script))  # type: ignore[no-untyped-call]


def image(
    explanation: Any,  # noqa: ANN401
    pixel_values: NDArray[Any] | None = None,
    labels: Sequence[str] | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | None:
    """Overlay attributions on the images they explain.

    The original image is shown desaturated underneath a red/blue attribution
    layer so positive and negative evidence stays readable.

    Parameters
    ----------
    explanation :
        An image Explanation or XWhy result.
    pixel_values :
        Optional images drawn underneath the attributions. If omitted,
        ``explanation.data`` is used.
    labels :
        Optional per-column titles, one per explained output.
    show :
        Whether to display the figure.
    save_path :
        Optional path to write the figure to.
    title :
        Optional figure title.
    figsize :
        Optional matplotlib figure size in inches (width drives layout).
    **kwargs :
        Extra options forwarded to :func:`_shap_image`
        (``true_labels``, ``width``, ``aspect``, ``hspace``, ``labelpad``,
        ``cmap``, ``vmax``).

    Returns
    -------
    Figure | None
        The figure when the host finish-helper returns it, otherwise
        ``None`` when ``show`` is True.

    """
    exp = _as_explanation(explanation) if _as_explanation is not None else explanation

    width = figsize[0] if figsize else kwargs.get("width", 20)
    fig = _shap_image(
        shap_values=exp,
        pixel_values=pixel_values,
        labels=labels,
        true_labels=kwargs.get("true_labels"),
        width=float(width),
        aspect=float(kwargs.get("aspect", 0.2)),
        hspace=kwargs.get("hspace", 0.2),
        labelpad=kwargs.get("labelpad"),
        cmap=kwargs.get("cmap", RED_TRANSPARENT_BLUE),
        vmax=kwargs.get("vmax"),
        show=False,
    )

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    if _finish_matplotlib is not None:
        return _finish_matplotlib(fig, show=show, save_path=save_path)
    if show:
        plt.show()
        return None
    return fig


def image_to_text(
    explanation: Any,  # noqa: ANN401
    *,
    max_tokens: int = 8,
    show: bool = True,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Display an interactive image-to-text attribution view.

    Args:
        explanation: Image Explanation or XWhy result with token outputs.
        max_tokens: Reserved for future token truncation (currently unused).
        show: Reserved for API compatibility (IPython always displays).
        save_path: Reserved; HTML view is not written to disk.
        title: Reserved for API compatibility.
        figsize: Reserved for API compatibility.
        **kwargs: Accepted for call compatibility; currently ignored.

    """
    _ = max_tokens, show, save_path, title, figsize, kwargs
    exp = _as_explanation(explanation) if _as_explanation is not None else explanation
    _shap_image_to_text(exp)
