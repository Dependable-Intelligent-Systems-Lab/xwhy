"""Native visualization engine for XWhy explanation results.

This module is a self-contained reimplementation of the plotting surface that
XWhy previously delegated to ``shap``. Every public function mirrors the SHAP
call signature it replaces, so existing notebooks keep working, but nothing
here imports ``shap``.

Design notes:
    * **matplotlib** is the default backend for every static figure.
    * **plotly** is offered on the plots where interactivity pays off, via
      ``backend="plotly"``.
    * **HTML** replaces SHAP's JavaScript bundles. :func:`text` and
      :func:`force` return plain, self-contained HTML strings, so there is no
      ``initjs`` handshake and no ``bundle.js`` to load. They render inline in
      notebooks and can be written straight to disk with ``save_path=...``.

Every plotting function follows the same output convention:
    * ``save_path`` given: the figure is written to disk and ``None`` returned.
    * ``show=True`` (default): the figure is displayed and ``None`` returned.
    * ``show=False``: the figure object is returned for further composition.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from numpy.typing import NDArray

# ==============================================================================
# PALETTE
# ==============================================================================

RED = "#ff0051"
BLUE = "#008afb"
GRAY = "#777777"
LIGHT_GRAY = "#c8c8c8"

BLUE_RGB: NDArray[Any] = np.array([0.0, 0.54337757, 0.98337906])
LIGHT_BLUE_RGB: NDArray[Any] = np.array(
    [0.4980392156862745, 0.7686274509803922, 0.9882352941176471]
)
RED_RGB: NDArray[Any] = np.array([1.0, 0.0, 0.31796406298163893])

SHAP_RED_BLUE_COLORS = [
    (0.0000, 0.5434, 0.9834, 1.0000),
    (0.0000, 0.5392, 0.9823, 1.0000),
    (0.0000, 0.5329, 0.9805, 1.0000),
    (0.0000, 0.5286, 0.9792, 1.0000),
    (0.0000, 0.5222, 0.9770, 1.0000),
    (0.0000, 0.5178, 0.9753, 1.0000),
    (0.0000, 0.5112, 0.9726, 1.0000),
    (0.0000, 0.5045, 0.9697, 1.0000),
    (0.0000, 0.5000, 0.9676, 1.0000),
    (0.0000, 0.4931, 0.9642, 1.0000),
    (0.0000, 0.4885, 0.9617, 1.0000),
    (0.0000, 0.4815, 0.9578, 1.0000),
    (0.0000, 0.4743, 0.9537, 1.0000),
    (0.0000, 0.4695, 0.9507, 1.0000),
    (0.0000, 0.4622, 0.9461, 1.0000),
    (0.0000, 0.4572, 0.9429, 1.0000),
    (0.0000, 0.4497, 0.9378, 1.0000),
    (0.0000, 0.4446, 0.9342, 1.0000),
    (0.0000, 0.4369, 0.9287, 1.0000),
    (0.0964, 0.4291, 0.9229, 1.0000),
    (0.1390, 0.4238, 0.9188, 1.0000),
    (0.1884, 0.4157, 0.9125, 1.0000),
    (0.2141, 0.4102, 0.9081, 1.0000),
    (0.2481, 0.4019, 0.9014, 1.0000),
    (0.2775, 0.3935, 0.8943, 1.0000),
    (0.2953, 0.3878, 0.8895, 1.0000),
    (0.3201, 0.3790, 0.8820, 1.0000),
    (0.3354, 0.3731, 0.8768, 1.0000),
    (0.3571, 0.3641, 0.8689, 1.0000),
    (0.3706, 0.3580, 0.8635, 1.0000),
    (0.3900, 0.3487, 0.8551, 1.0000),
    (0.4083, 0.3392, 0.8464, 1.0000),
    (0.4199, 0.3327, 0.8405, 1.0000),
    (0.4367, 0.3228, 0.8314, 1.0000),
    (0.4473, 0.3160, 0.8252, 1.0000),
    (0.4628, 0.3057, 0.8157, 1.0000),
    (0.4774, 0.2952, 0.8060, 1.0000),
    (0.4869, 0.2879, 0.7994, 1.0000),
    (0.5005, 0.2768, 0.7892, 1.0000),
    (0.5093, 0.2692, 0.7823, 1.0000),
    (0.5220, 0.2575, 0.7718, 1.0000),
    (0.5341, 0.2454, 0.7610, 1.0000),
    (0.5420, 0.2370, 0.7537, 1.0000),
    (0.5533, 0.2240, 0.7425, 1.0000),
    (0.5605, 0.2150, 0.7350, 1.0000),
    (0.5711, 0.2010, 0.7235, 1.0000),
    (0.5778, 0.1911, 0.7157, 1.0000),
    (0.5876, 0.1755, 0.7039, 1.0000),
    (0.5969, 0.1587, 0.6918, 1.0000),
    (0.6029, 0.1467, 0.6837, 1.0000),
    (0.6153, 0.1319, 0.6749, 1.0000),
    (0.6258, 0.1246, 0.6713, 1.0000),
    (0.6412, 0.1128, 0.6658, 1.0000),
    (0.6563, 0.0996, 0.6600, 1.0000),
    (0.6663, 0.0900, 0.6560, 1.0000),
    (0.6808, 0.0735, 0.6499, 1.0000),
    (0.6904, 0.0609, 0.6457, 1.0000),
    (0.7044, 0.0376, 0.6392, 1.0000),
    (0.7136, 0.0198, 0.6348, 1.0000),
    (0.7271, 0.0000, 0.6281, 1.0000),
    (0.7403, 0.0000, 0.6211, 1.0000),
    (0.7490, 0.0000, 0.6164, 1.0000),
    (0.7617, 0.0000, 0.6091, 1.0000),
    (0.7700, 0.0000, 0.6042, 1.0000),
    (0.7823, 0.0000, 0.5966, 1.0000),
    (0.7942, 0.0000, 0.5889, 1.0000),
    (0.8020, 0.0000, 0.5837, 1.0000),
    (0.8135, 0.0000, 0.5758, 1.0000),
    (0.8210, 0.0000, 0.5704, 1.0000),
    (0.8319, 0.0000, 0.5622, 1.0000),
    (0.8426, 0.0000, 0.5539, 1.0000),
    (0.8496, 0.0000, 0.5482, 1.0000),
    (0.8597, 0.0000, 0.5397, 1.0000),
    (0.8664, 0.0000, 0.5339, 1.0000),
    (0.8761, 0.0000, 0.5252, 1.0000),
    (0.8824, 0.0000, 0.5193, 1.0000),
    (0.8916, 0.0000, 0.5103, 1.0000),
    (0.9006, 0.0000, 0.5013, 1.0000),
    (0.9064, 0.0000, 0.4952, 1.0000),
    (0.9148, 0.0000, 0.4860, 1.0000),
    (0.9203, 0.0000, 0.4798, 1.0000),
    (0.9283, 0.0000, 0.4705, 1.0000),
    (0.9359, 0.0000, 0.4610, 1.0000),
    (0.9409, 0.0000, 0.4547, 1.0000),
    (0.9481, 0.0000, 0.4451, 1.0000),
    (0.9527, 0.0000, 0.4387, 1.0000),
    (0.9594, 0.0000, 0.4290, 1.0000),
    (0.9637, 0.0000, 0.4225, 1.0000),
    (0.9700, 0.0000, 0.4126, 1.0000),
    (0.9759, 0.0000, 0.4028, 1.0000),
    (0.9797, 0.0000, 0.3961, 1.0000),
    (0.9851, 0.0000, 0.3862, 1.0000),
    (0.9886, 0.0000, 0.3795, 1.0000),
    (0.9935, 0.0000, 0.3694, 1.0000),
    (0.9979, 0.0000, 0.3592, 1.0000),
    (1.0000, 0.0000, 0.3524, 1.0000),
    (1.0000, 0.0000, 0.3422, 1.0000),
    (1.0000, 0.0000, 0.3353, 1.0000),
    (1.0000, 0.0000, 0.3249, 1.0000),
    (1.0000, 0.0000, 0.3180, 1.0000),
]
RED_BLUE = LinearSegmentedColormap.from_list(
    "xwhy_red_blue", SHAP_RED_BLUE_COLORS, N=256
)

SHAP_RED_BLUE_CIRCLE_COLORS = [
    (0.0000, 0.5434, 0.9834, 1.0000),
    (0.0000, 0.5394, 0.9863, 1.0000),
    (0.0000, 0.5332, 0.9897, 1.0000),
    (0.0000, 0.5288, 0.9914, 1.0000),
    (0.0000, 0.5218, 0.9931, 1.0000),
    (0.0000, 0.5169, 0.9936, 1.0000),
    (0.0000, 0.5092, 0.9934, 1.0000),
    (0.0000, 0.5010, 0.9920, 1.0000),
    (0.0125, 0.4953, 0.9905, 1.0000),
    (0.1737, 0.4863, 0.9872, 1.0000),
    (0.2385, 0.4800, 0.9844, 1.0000),
    (0.3116, 0.4702, 0.9792, 1.0000),
    (0.3710, 0.4598, 0.9729, 1.0000),
    (0.4058, 0.4526, 0.9680, 1.0000),
    (0.4536, 0.4414, 0.9597, 1.0000),
    (0.4830, 0.4335, 0.9536, 1.0000),
    (0.5243, 0.4213, 0.9435, 1.0000),
    (0.5502, 0.4128, 0.9361, 1.0000),
    (0.5870, 0.3996, 0.9241, 1.0000),
    (0.6217, 0.3858, 0.9111, 1.0000),
    (0.6437, 0.3762, 0.9018, 1.0000),
    (0.6753, 0.3614, 0.8870, 1.0000),
    (0.6953, 0.3511, 0.8766, 1.0000),
    (0.7242, 0.3351, 0.8601, 1.0000),
    (0.7515, 0.3183, 0.8428, 1.0000),
    (0.7689, 0.3067, 0.8307, 1.0000),
    (0.7939, 0.2887, 0.8118, 1.0000),
    (0.8097, 0.2762, 0.7988, 1.0000),
    (0.8324, 0.2567, 0.7786, 1.0000),
    (0.8467, 0.2432, 0.7646, 1.0000),
    (0.8671, 0.2219, 0.7432, 1.0000),
    (0.8861, 0.1993, 0.7210, 1.0000),
    (0.8981, 0.1834, 0.7059, 1.0000),
    (0.9149, 0.1578, 0.6828, 1.0000),
    (0.9254, 0.1393, 0.6670, 1.0000),
    (0.9401, 0.1084, 0.6429, 1.0000),
    (0.9534, 0.0705, 0.6184, 1.0000),
    (0.9616, 0.0366, 0.6019, 1.0000),
    (0.9727, 0.0000, 0.5767, 1.0000),
    (0.9794, 0.0000, 0.5597, 1.0000),
    (0.9884, 0.0000, 0.5341, 1.0000),
    (0.9961, 0.0000, 0.5081, 1.0000),
    (0.9997, 0.0000, 0.4908, 1.0000),
    (1.0000, 0.0000, 0.4645, 1.0000),
    (1.0000, 0.0000, 0.4469, 1.0000),
    (1.0000, 0.0000, 0.4205, 1.0000),
    (1.0000, 0.0000, 0.4028, 1.0000),
    (1.0000, 0.0000, 0.3761, 1.0000),
    (1.0000, 0.0000, 0.3493, 1.0000),
    (1.0000, 0.0000, 0.3314, 1.0000),
    (1.0000, 0.0048, 0.2901, 1.0000),
    (0.9966, 0.0703, 0.2532, 1.0000),
    (0.9781, 0.1569, 0.1973, 1.0000),
    (0.9544, 0.2151, 0.1374, 1.0000),
    (0.9365, 0.2482, 0.0912, 1.0000),
    (0.9065, 0.2919, 0.0000, 1.0000),
    (0.8847, 0.3181, 0.0000, 1.0000),
    (0.8496, 0.3540, 0.0000, 1.0000),
    (0.8246, 0.3758, 0.0000, 1.0000),
    (0.7851, 0.4059, 0.0000, 1.0000),
    (0.7434, 0.4329, 0.0000, 1.0000),
    (0.7146, 0.4494, 0.0000, 1.0000),
    (0.6697, 0.4720, 0.0000, 1.0000),
    (0.6388, 0.4858, 0.0000, 1.0000),
    (0.5909, 0.5046, 0.0000, 1.0000),
    (0.5411, 0.5214, 0.0000, 1.0000),
    (0.5069, 0.5315, 0.0000, 1.0000),
    (0.4533, 0.5451, 0.0000, 1.0000),
    (0.4158, 0.5533, 0.0000, 1.0000),
    (0.3554, 0.5643, 0.0000, 1.0000),
    (0.2870, 0.5739, 0.0000, 1.0000),
    (0.2335, 0.5796, 0.0000, 1.0000),
    (0.1144, 0.5871, 0.0000, 1.0000),
    (0.0017, 0.5915, 0.0371, 1.0000),
    (0.0000, 0.5973, 0.1334, 1.0000),
    (0.0000, 0.6007, 0.1793, 1.0000),
    (0.0000, 0.6051, 0.2405, 1.0000),
    (0.0000, 0.6087, 0.2977, 1.0000),
    (0.0000, 0.6108, 0.3349, 1.0000),
    (0.0000, 0.6134, 0.3899, 1.0000),
    (0.0000, 0.6148, 0.4262, 1.0000),
    (0.0000, 0.6164, 0.4800, 1.0000),
    (0.0000, 0.6175, 0.5331, 1.0000),
    (0.0000, 0.6179, 0.5679, 1.0000),
    (0.0000, 0.6181, 0.6189, 1.0000),
    (0.0000, 0.6178, 0.6520, 1.0000),
    (0.0000, 0.6169, 0.6999, 1.0000),
    (0.0000, 0.6160, 0.7306, 1.0000),
    (0.0000, 0.6139, 0.7742, 1.0000),
    (0.0000, 0.6110, 0.8146, 1.0000),
    (0.0000, 0.6085, 0.8396, 1.0000),
    (0.0000, 0.6040, 0.8737, 1.0000),
    (0.0000, 0.6005, 0.8943, 1.0000),
    (0.0000, 0.5942, 0.9212, 1.0000),
    (0.0000, 0.5867, 0.9436, 1.0000),
    (0.0000, 0.5810, 0.9559, 1.0000),
    (0.0000, 0.5713, 0.9700, 1.0000),
    (0.0000, 0.5641, 0.9767, 1.0000),
    (0.0000, 0.5521, 0.9823, 1.0000),
    (0.0000, 0.5434, 0.9834, 1.0000),
]
RED_BLUE_CIRCLE = LinearSegmentedColormap.from_list(
    "xwhy_red_blue_circle", SHAP_RED_BLUE_CIRCLE_COLORS, N=256
)

SHAP_RED_WHITE_BLUE_COLORS = [
    (0.0000, 0.5434, 0.9834, 1.0000),
    (0.0158, 0.5506, 0.9836, 1.0000),
    (0.0394, 0.5614, 0.9840, 1.0000),
    (0.0552, 0.5686, 0.9843, 1.0000),
    (0.0788, 0.5794, 0.9847, 1.0000),
    (0.0946, 0.5866, 0.9850, 1.0000),
    (0.1182, 0.5974, 0.9853, 1.0000),
    (0.1419, 0.6082, 0.9857, 1.0000),
    (0.1577, 0.6154, 0.9860, 1.0000),
    (0.1813, 0.6262, 0.9864, 1.0000),
    (0.1971, 0.6334, 0.9867, 1.0000),
    (0.2207, 0.6442, 0.9870, 1.0000),
    (0.2444, 0.6550, 0.9874, 1.0000),
    (0.2601, 0.6622, 0.9877, 1.0000),
    (0.2838, 0.6730, 0.9881, 1.0000),
    (0.2995, 0.6802, 0.9884, 1.0000),
    (0.3232, 0.6910, 0.9888, 1.0000),
    (0.3390, 0.6982, 0.9890, 1.0000),
    (0.3626, 0.7090, 0.9894, 1.0000),
    (0.3863, 0.7198, 0.9898, 1.0000),
    (0.4020, 0.7269, 0.9901, 1.0000),
    (0.4257, 0.7377, 0.9905, 1.0000),
    (0.4414, 0.7449, 0.9907, 1.0000),
    (0.4651, 0.7557, 0.9911, 1.0000),
    (0.4887, 0.7665, 0.9915, 1.0000),
    (0.5045, 0.7737, 0.9918, 1.0000),
    (0.5281, 0.7845, 0.9922, 1.0000),
    (0.5439, 0.7917, 0.9924, 1.0000),
    (0.5676, 0.8025, 0.9928, 1.0000),
    (0.5833, 0.8097, 0.9931, 1.0000),
    (0.6070, 0.8205, 0.9935, 1.0000),
    (0.6306, 0.8313, 0.9939, 1.0000),
    (0.6464, 0.8385, 0.9941, 1.0000),
    (0.6700, 0.8493, 0.9945, 1.0000),
    (0.6858, 0.8565, 0.9948, 1.0000),
    (0.7094, 0.8673, 0.9952, 1.0000),
    (0.7331, 0.8781, 0.9956, 1.0000),
    (0.7489, 0.8853, 0.9958, 1.0000),
    (0.7725, 0.8961, 0.9962, 1.0000),
    (0.7883, 0.9033, 0.9965, 1.0000),
    (0.8119, 0.9141, 0.9969, 1.0000),
    (0.8356, 0.9249, 0.9973, 1.0000),
    (0.8513, 0.9321, 0.9975, 1.0000),
    (0.8750, 0.9429, 0.9979, 1.0000),
    (0.8908, 0.9501, 0.9982, 1.0000),
    (0.9144, 0.9609, 0.9986, 1.0000),
    (0.9302, 0.9681, 0.9988, 1.0000),
    (0.9538, 0.9789, 0.9992, 1.0000),
    (0.9775, 0.9897, 0.9996, 1.0000),
    (0.9932, 0.9969, 0.9999, 1.0000),
    (1.0000, 0.9932, 0.9954, 1.0000),
    (1.0000, 0.9775, 0.9846, 1.0000),
    (1.0000, 0.9538, 0.9685, 1.0000),
    (1.0000, 0.9302, 0.9524, 1.0000),
    (1.0000, 0.9144, 0.9416, 1.0000),
    (1.0000, 0.8908, 0.9255, 1.0000),
    (1.0000, 0.8750, 0.9147, 1.0000),
    (1.0000, 0.8513, 0.8986, 1.0000),
    (1.0000, 0.8356, 0.8879, 1.0000),
    (1.0000, 0.8119, 0.8717, 1.0000),
    (1.0000, 0.7883, 0.8556, 1.0000),
    (1.0000, 0.7725, 0.8448, 1.0000),
    (1.0000, 0.7489, 0.8287, 1.0000),
    (1.0000, 0.7331, 0.8180, 1.0000),
    (1.0000, 0.7094, 0.8018, 1.0000),
    (1.0000, 0.6858, 0.7857, 1.0000),
    (1.0000, 0.6700, 0.7750, 1.0000),
    (1.0000, 0.6464, 0.7588, 1.0000),
    (1.0000, 0.6306, 0.7481, 1.0000),
    (1.0000, 0.6070, 0.7319, 1.0000),
    (1.0000, 0.5833, 0.7158, 1.0000),
    (1.0000, 0.5676, 0.7051, 1.0000),
    (1.0000, 0.5439, 0.6889, 1.0000),
    (1.0000, 0.5281, 0.6782, 1.0000),
    (1.0000, 0.5045, 0.6620, 1.0000),
    (1.0000, 0.4887, 0.6513, 1.0000),
    (1.0000, 0.4651, 0.6352, 1.0000),
    (1.0000, 0.4414, 0.6190, 1.0000),
    (1.0000, 0.4257, 0.6083, 1.0000),
    (1.0000, 0.4020, 0.5922, 1.0000),
    (1.0000, 0.3863, 0.5814, 1.0000),
    (1.0000, 0.3626, 0.5653, 1.0000),
    (1.0000, 0.3390, 0.5491, 1.0000),
    (1.0000, 0.3232, 0.5384, 1.0000),
    (1.0000, 0.2995, 0.5223, 1.0000),
    (1.0000, 0.2838, 0.5115, 1.0000),
    (1.0000, 0.2601, 0.4954, 1.0000),
    (1.0000, 0.2444, 0.4846, 1.0000),
    (1.0000, 0.2207, 0.4685, 1.0000),
    (1.0000, 0.1971, 0.4524, 1.0000),
    (1.0000, 0.1813, 0.4416, 1.0000),
    (1.0000, 0.1577, 0.4255, 1.0000),
    (1.0000, 0.1419, 0.4147, 1.0000),
    (1.0000, 0.1182, 0.3986, 1.0000),
    (1.0000, 0.0946, 0.3825, 1.0000),
    (1.0000, 0.0788, 0.3717, 1.0000),
    (1.0000, 0.0552, 0.3556, 1.0000),
    (1.0000, 0.0394, 0.3448, 1.0000),
    (1.0000, 0.0158, 0.3287, 1.0000),
    (1.0000, 0.0000, 0.3180, 1.0000),
]
RED_WHITE_BLUE = LinearSegmentedColormap.from_list(
    "xwhy_red_white_blue", SHAP_RED_WHITE_BLUE_COLORS, N=256
)

RED_TRANSPARENT_BLUE = LinearSegmentedColormap.from_list(
    "xwhy_red_transparent_blue",
    [
        (0.0, (0.1176, 0.5333, 0.8980, 1.0)),
        (0.5, (0.1176, 0.5333, 0.8980, 0.0)),
        (0.5000000001, (1.0, 0.0510, 0.3412, 0.0)),
        (1.0, (1.0, 0.0510, 0.3412, 1.0)),
    ],
    N=256,
)

PLOTLY_RED_BLUE = [
    [0.0, BLUE],
    [0.5, LIGHT_GRAY],
    [1.0, RED],
]

#: Axis label used wherever SHAP would have written "XWhy value".
VALUE_LABEL = "XWhy value"

_MATPLOTLIB_BACKENDS = frozenset({"matplotlib", "mpl"})
_PLOTLY_BACKENDS = frozenset({"plotly", "px", "go"})
_HTML_BACKENDS = frozenset({"html"})

# ==============================================================================
# EXPLANATION CONTAINER
# ==============================================================================


@dataclass
class Explanation:
    """Container for attribution values, mirroring ``shap.Explanation``.

    This is the interchange format between :mod:`xwhy.core.result` and this
    module. It supports the slicing and reduction idioms that SHAP users
    expect (``exp[0]``, ``exp[:, 2]``, ``exp.abs.mean(0)``) without depending
    on ``shap``.

    Attributes:
        values: Attribution values. Shape is ``(n_features,)`` for a single
            explained instance, ``(n_instances, n_features)`` for a batch, or
            higher-rank for image and multimodal explanations.
        base_values: The model's expected output, i.e. the value the
            attributions are measured against.
        data: The underlying instance(s) the attributions describe.
        feature_names: Names aligned with the last axis of ``values``.
        display_data: Optional human-readable stand-in for ``data``.
        output_names: Optional names for the model outputs.

    """

    values: np.ndarray
    base_values: float | np.ndarray = 0.0
    data: np.ndarray | Sequence[Any] | None = None
    feature_names: Sequence[str] | np.ndarray | None = None
    display_data: np.ndarray | None = None
    output_names: Sequence[str] | None = None
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None
    error_std: Any | None = None
    clustering: np.ndarray | None = None
    hierarchical_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Coerce ``values`` to a numpy array so downstream maths is safe."""
        self.values = np.asarray(self.values)

    # -- shape helpers ---------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying attribution array."""
        return cast(tuple[int, ...], self.values.shape)

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying attribution array."""
        return int(self.values.ndim)

    def __len__(self) -> int:
        """Return the size of the leading axis."""
        return int(self.values.shape[0]) if self.values.ndim else 0

    def __getitem__(self, item: Any) -> Explanation:  # noqa: ANN401
        """Slice the explanation, keeping metadata aligned.

        Args:
            item: Any numpy-compatible index. A tuple index whose second
                element selects features also slices ``feature_names``.

        Returns:
            Explanation: A new explanation covering the selected subset.

        """
        names = self.feature_names
        data = self.data

        if isinstance(item, tuple) and len(item) > 1 and names is not None:
            names = np.asarray(names)[item[1]]

        if data is not None:
            data_arr = np.asarray(data, dtype=object)
            try:
                data = data_arr[item]
            except (IndexError, ValueError):
                data = None

        base = self.base_values
        if isinstance(base, np.ndarray) and base.ndim > 0:
            try:
                base = base[item[0] if isinstance(item, tuple) else item]
            except (IndexError, ValueError):
                base = self.base_values

        return Explanation(
            values=self.values[item],
            base_values=base,
            data=data,
            feature_names=names,
            output_names=self.output_names,
        )

    # -- reductions ------------------------------------------------------

    @property
    def abs(self) -> Explanation:
        """Explanation holding the absolute value of every attribution."""
        return Explanation(
            values=np.abs(self.values),
            base_values=self.base_values,
            data=self.data,
            feature_names=self.feature_names,
            output_names=self.output_names,
        )

    def _reduce(self, func: Callable[..., Any], axis: int | None) -> Explanation:
        """Apply a numpy reduction while preserving feature metadata."""
        reduced = func(self.values, axis=axis)
        keeps_features = axis == 0 and self.values.ndim > 1
        return Explanation(
            values=np.asarray(reduced),
            base_values=self.base_values,
            feature_names=self.feature_names if keeps_features else None,
            output_names=self.output_names,
        )

    def mean(self, axis: int | None = None) -> Explanation:
        """Return the mean of the attributions along ``axis``."""
        return self._reduce(np.mean, axis)

    def sum(self, axis: int | None = None) -> Explanation:
        """Return the sum of the attributions along ``axis``."""
        return self._reduce(np.sum, axis)

    def max(self, axis: int | None = None) -> Explanation:
        """Return the maximum of the attributions along ``axis``."""
        return self._reduce(np.max, axis)

    def min(self, axis: int | None = None) -> Explanation:
        """Return the minimum of the attributions along ``axis``."""
        return self._reduce(np.min, axis)

    def __repr__(self) -> str:
        """Return a short, informative representation."""
        return (
            f"Explanation(values={self.values.shape}, "
            f"base_values={np.shape(self.base_values)}, "
            f"n_names={0 if self.feature_names is None else len(self.feature_names)})"
        )


class DimensionError(Exception):
    """Raised when XWhy value and feature matrix shapes are incompatible."""


# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================


def _as_explanation(obj: Any) -> Explanation:  # noqa: ANN401
    """Normalise an explanation-like object into an :class:`Explanation`.

    Args:
        obj: An :class:`Explanation`, an object exposing ``to_explanation()``
            or ``to_shap()`` (e.g. a ``BaseXWhyResult``), or a raw array.

    Returns:
        Explanation: The normalised explanation.

    Raises:
        TypeError: If the object cannot be interpreted as an explanation.

    """
    if isinstance(obj, Explanation):
        return obj

    if type(obj).__name__ == "Explanation" and hasattr(obj, "values"):
        return Explanation(
            values=np.asarray(obj.values),
            base_values=getattr(obj, "base_values", 0.0),
            data=getattr(obj, "data", None),
            feature_names=getattr(obj, "feature_names", None),
            display_data=getattr(obj, "display_data", None),
            output_names=getattr(obj, "output_names", None),
        )

    for attr in ("to_explanation", "to_shap"):
        converter = getattr(obj, attr, None)
        if callable(converter):
            converted = converter()
            if isinstance(converted, Explanation):
                return converted
            # A foreign Explanation-like object: copy the fields we need.
            return Explanation(
                values=np.asarray(getattr(converted, "values", converted)),
                base_values=getattr(converted, "base_values", 0.0),
                data=getattr(converted, "data", None),
                feature_names=getattr(converted, "feature_names", None),
            )

    if isinstance(obj, np.ndarray):
        return Explanation(values=obj)

    raise TypeError(
        f"Cannot build an Explanation from {type(obj).__name__}. Pass an "
        "Explanation, an XWhy result, or a numpy array."
    )


def _resolve_names(
    names: Sequence[str] | np.ndarray | None,
    n_features: int,
) -> list[str]:
    """Return exactly ``n_features`` display names, generating any that are missing."""
    if names is None:
        return [f"Feature {i}" for i in range(n_features)]

    resolved = [str(n) for n in np.asarray(names).ravel().tolist()]
    if len(resolved) >= n_features:
        return resolved[:n_features]

    resolved.extend(f"Feature {i}" for i in range(len(resolved), n_features))
    return resolved


def _format_value(value: Any, fmt: str = "%.2f") -> str:  # noqa: ANN401
    """Format a feature value for display, trimming redundant trailing zeros."""
    if value is None:
        return ""
    if isinstance(value, (str, np.str_)):
        return str(value)
    try:
        formatted = fmt % float(value)
    except (TypeError, ValueError):
        return str(value)

    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return "0" if formatted in {"-0", "", "+0"} else formatted


def _global_importance(values: np.ndarray) -> np.ndarray:
    """Collapse an attribution array to one importance score per feature."""
    if values.ndim == 1:
        return np.abs(values)
    axes = tuple(range(values.ndim - 1))
    return np.abs(values).mean(axis=axes)


def _group_minor_features(
    values: np.ndarray,
    names: Sequence[str],
    max_display: int | None,
) -> tuple[np.ndarray, list[str]]:
    """Keep the strongest features and fold the remainder into one summary row.

    Args:
        values: One score per feature.
        names: Feature names aligned with ``values``.
        max_display: Maximum number of rows to draw, including the summary
            row. ``None`` disables grouping.

    Returns:
        tuple: ``(values, names)`` sorted ascending by magnitude, ready for a
        bottom-to-top horizontal bar chart.

    """
    values = np.asarray(values, dtype=float).ravel()
    names = list(names)
    order = np.argsort(np.abs(values))[::-1]

    if max_display is not None and 0 < max_display < len(order):
        keep, rest = order[: max_display - 1], order[max_display - 1 :]
        kept_values = np.append(values[keep], values[rest].sum())
        kept_names = [names[i] for i in keep]
        kept_names.append(f"Sum of {len(rest)} other features")
    else:
        kept_values = values[order]
        kept_names = [names[i] for i in order]

    return kept_values[::-1], kept_names[::-1]


def _check_backend(backend: str, allowed: frozenset[str]) -> str:
    """Validate a backend name and return its canonical form.

    Args:
        backend: User-supplied backend name.
        allowed: The set of canonical backends this plot supports.

    Returns:
        str: One of ``"matplotlib"``, ``"plotly"`` or ``"html"``.

    Raises:
        ValueError: If the backend is unknown or unsupported for this plot.

    """
    key = backend.lower().strip()

    if key in _MATPLOTLIB_BACKENDS:
        canonical = "matplotlib"
    elif key in _PLOTLY_BACKENDS:
        canonical = "plotly"
    elif key in _HTML_BACKENDS:
        canonical = "html"
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose from 'matplotlib', "
            "'plotly' or 'html'."
        )

    if canonical not in allowed:
        raise ValueError(
            f"Backend {backend!r} is not supported by this plot. "
            f"Supported backends: {', '.join(sorted(allowed))}."
        )
    return canonical


def _finish_matplotlib(
    fig: Figure,
    *,
    show: bool,
    save_path: str | Path | None,
) -> Figure | None:
    """Save, display or hand back a matplotlib figure."""
    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return None

    if show:
        plt.show()
        return None

    return fig


def _finish_plotly(
    fig: go.Figure,
    *,
    show: bool,
    save_path: str | Path | None,
) -> go.Figure | None:
    """Save, display or hand back a plotly figure."""
    if save_path is not None:
        path_str = str(save_path)
        if path_str.endswith(".html"):
            fig.write_html(path_str)
        else:
            fig.write_image(path_str)
        return None

    if show:
        fig.show()
        return None

    return fig


def _finish_html(
    html: str,
    *,
    show: bool,
    save_path: str | Path | None,
) -> str:
    """Save and/or display an HTML fragment, always returning the markup."""
    if save_path is not None:
        Path(save_path).write_text(_wrap_html_document(html), encoding="utf-8")
        return html

    if show:
        _display_html(html)

    return html


def _display_html(html: str) -> None:
    """Render HTML inline when running inside IPython, otherwise do nothing."""
    try:  # pragma: no cover - depends on the runtime environment
        from IPython.display import HTML, display
    except ImportError:  # pragma: no cover - plain interpreter
        return

    display(HTML(html))  # type: ignore[no-untyped-call]


def _wrap_html_document(fragment: str, title: str = "XWhy explanation") -> str:
    """Wrap an HTML fragment in a minimal standalone document."""
    return (
        "<!doctype html>\n<html><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title></head><body>{fragment}</body></html>"
    )


def _style_axes(ax: Axes) -> None:
    """Apply the shared XWhy axis styling (minimal chrome, soft gridlines)."""
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(visible=True, color="#eeeeee", linewidth=0.8)


def _single_instance(exp: Explanation) -> tuple[np.ndarray, float, list[str], Any]:
    """Reduce an explanation to one instance for the local plots.

    Args:
        exp: The explanation to reduce.

    Returns:
        tuple: ``(values, base_value, feature_names, data)`` for one instance.

    Raises:
        ValueError: If the explanation holds more than one instance.

    """
    values = np.asarray(exp.values, dtype=float)
    data = exp.data
    base = exp.base_values

    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
        if data is not None and len(np.shape(data)) == 2:
            data = np.asarray(data, dtype=object)[0]
        if isinstance(base, np.ndarray) and base.size == 1:
            base = float(base.reshape(-1)[0])
    elif values.ndim > 1:
        raise ValueError(
            "This plot explains a single instance but received values with "
            f"shape {values.shape}. Index the explanation first, e.g. exp[0]."
        )

    if isinstance(base, np.ndarray):
        base = float(base.reshape(-1)[0]) if base.size else 0.0

    names = _resolve_names(exp.feature_names, values.shape[0])
    return values, float(base), names, data


def convert_name(
    ind: str | int | None,
    shap_values: npt.NDArray[Any] | None,
    input_names: list[str] | npt.NDArray[Any] | None,
) -> int | str | None:
    """Map a feature name, rank expression, or index to a column index."""
    if ind is None:
        return None
    if not isinstance(ind, str):
        return ind

    names_arr = np.array(input_names) if input_names is not None else np.array([])
    nzinds = np.where(names_arr == ind)[0]
    if len(nzinds) == 0:
        if ind.startswith("rank("):
            if shap_values is None:
                msg = "shap_values must be provided for rank-based indexing"
                raise ValueError(msg)
            rank = int(ind[5:-1])
            return int(np.argsort(-np.abs(shap_values).mean(0))[rank])
        if ind == "sum()":
            return "sum()"
        msg = f"Could not find feature named: {ind}"
        raise ValueError(msg)
    return int(nzinds[0])


def initjs() -> None:
    """No-op kept so SHAP-style notebooks keep running unchanged.

    SHAP required this call to inject its JavaScript bundle before force
    and text plots would render. XWhy uses static HTML and matplotlib, so
    there is no JavaScript to initialise.
    """
    return None
