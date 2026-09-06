# Pix2Pix-style models

Pix2Pix-style models are conditional image-to-image systems: they transform a source image into a target image rather than returning a classification score. They are one useful model family for studying image-editing explainability, but they are **not a separate public XWhy explainer**.

Public XWhy component

The current public API is `ImageGenerationAndEditingExplainer`. There is no exported `Pix2PixExplainer` in the current package.

## Model-agnostic explanation workflow

For a compatible Pix2Pix-style model, the broader XWhy workflow can be applied by supplying the model through a supported pipeline, engine, or custom generation/editing function. A local explanation can then involve:

1. producing a reference transformation from the original source image and instruction or conditioning input;
1. perturbing the textual conditioning input;
1. rerunning the image transformation for each perturbation;
1. comparing perturbed outputs with the reference output;
1. measuring image differences in pixel or embedding space;
1. weighting the local neighbourhood using text similarity or distance; and
1. fitting an interpretable local surrogate to estimate term contributions.

## Why Pix2Pix remains useful as an example

Pix2Pix-style models provide a comparatively clear source-to-output relationship and are therefore useful for evaluating explanation methods under controlled image-to-image transformations. A reproducible example should document:

- the source and target domains;
- model weights and preprocessing;
- the conditioning or editing instruction, where applicable;
- perturbation strategy;
- output-distance measure;
- local surrogate configuration;
- attribution results;
- fidelity and stability evidence; and
- known limitations.

## Scope

The current `ImageGenerationAndEditingExplainer` primarily attributes changes in output images to perturbations of the textual instruction. A future Pix2Pix-specific study could extend this to source-image region perturbation and direct region-level attribution while using the same local explanation and evaluation principles.

[Read the image generation and editing overview](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/image-generation/index.md)

[View the current image explainer API reference](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/xwhy/explainers/image/index.md)
