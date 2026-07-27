---
title: XWhy Pix2Pix Explainer
description: Experimental development status for explaining image-to-image and instruction-conditioned transformation models with XWhy.
---

# Pix2Pix explainer

!!! danger "Experimental interface — under construction"
    `Pix2PixExplainer` is exported by XWhy, but its current `explain()` method raises `NotImplementedError`. The public positioning of this component should be confirmed before release.

The reserved scope is image-to-image or instruction-conditioned transformation explanation.

Planned documentation will define:

- supported transformation models;
- input and output comparison units;
- image-region and instruction perturbations;
- output-distance measures;
- visual explanation formats;
- safety and misuse limitations.

[View the current API reference](../reference/xwhy/explainers/pix2pix/)
