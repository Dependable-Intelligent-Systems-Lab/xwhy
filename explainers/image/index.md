# Image classification explainer

Available

`ImageClassificationExplainer` is implemented and has a complete usage guide.

The image-classification explainer perturbs image regions, observes changes in model predictions, and estimates which regions have the strongest local influence.

Current documented capabilities include:

- built-in PyTorch classification models;
- custom PyTorch models and preprocessing;
- DINOv2 image embeddings;
- supported segmentation models or a supplied mask;
- multiple statistical distance measures;
- image and image-heatmap visualisations.

[Read the complete image-classification tutorial](https://dependable-intelligent-systems-lab.github.io/xwhy/image_classification_explainer/index.md)

[Open the image explainer API reference](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/xwhy/explainers/image/index.md)

## Scope

The currently implemented workflow is **image classification**. Broader image tasks should be described as planned until a corresponding implementation and test suite are added.
