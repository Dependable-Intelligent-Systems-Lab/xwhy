---
title: Image Classification Explainability with XWhy
description: Explain PyTorch image-classification predictions with XWhy using region perturbation, statistical distances, local surrogate models, and image heatmaps.
---

# Image classification explainer

!!! success "Available"
    `ImageClassificationExplainer` is implemented and has a complete usage guide.

The image-classification explainer perturbs image regions, observes changes in model predictions, and estimates which regions have the strongest local influence.

Current documented capabilities include:

- built-in PyTorch classification models;
- custom PyTorch models and preprocessing;
- DINOv2 image embeddings;
- supported segmentation models or a supplied mask;
- multiple statistical distance measures;
- image and image-heatmap visualisations.

[Read the complete image-classification tutorial](../../image_classification_explainer.md)

[Open the image explainer API reference](../../reference/xwhy/explainers/image.md)

## Scope

The currently implemented workflow is **image classification**. Broader image tasks should be described as planned until a corresponding implementation and test suite are added.
