---
title: Image Generation and Image Editing
description: An overview of model-agnostic explanation approaches for image-generation and image-editing models in XWhy.
---

# Image Generation and Image Editing

Generative image models either create new images or transform existing ones. Within XWhy, these tasks are organised into two related categories:

- **Image generation**, where a model creates an image from a conditioning input such as text, noise, or another representation.
- **Image editing**, where a source image is modified according to an instruction, mask, or target condition.

Unlike image classification, these models do not return a single class score. Their explanations must therefore consider the relationship between the input condition, the source image, and the generated output.

## Model-agnostic explanation workflow

A local explanation workflow for image-generation and image-editing models may involve:

1. generating a reference output using the original input;
2. perturbing the source image, conditioning input, or editing instruction;
3. generating outputs for the perturbed inputs;
4. comparing each perturbed output with the reference output;
5. measuring differences using pixel-space, perceptual, or semantic distances;
6. fitting a local interpretable surrogate model;
7. producing feature, region, or word-level attributions.

This approach can be applied without requiring access to the internal architecture or gradients of the generative model.

## Image-editing example: Pix2Pix

Pix2Pix-style models are an example of conditional image-to-image generation. They transform a source image into a target image rather than returning a classification score.

!!! danger "Experimental interface — under construction"
    XWhy currently exports `Pix2PixExplainer`, but its `explain()` method raises `NotImplementedError`. The interface is therefore experimental and should not yet be treated as an executable explanation workflow.

A future Pix2Pix example should document:

1. the source image and target transformation;
2. the model, weights, and preprocessing steps;
3. the source-image regions selected for perturbation;
4. the output-distance or perceptual-similarity measure;
5. the local surrogate configuration;
6. the resulting attribution map;
7. fidelity, stability, and runtime evidence;
8. limitations when interpreting generative outputs.

## Planned examples

Future documentation may include:

- text-to-image generation;
- instruction-based image editing;
- mask-based image editing;
- Pix2Pix-style image translation;
- perturbation of textual conditioning inputs;
- comparison of pixel, perceptual, and semantic distances;
- detection of unintended changes outside the requested edit region.

## Evaluation considerations

Explanations for generative models should be assessed across multiple dimensions, including:

- **fidelity** to the behaviour of the generative model;
- **stability** under small or semantically irrelevant input changes;
- **consistency** across repeated generations;
- **runtime and computational cost**;
- **localisation** of intended and unintended visual changes.

[View the current `Pix2PixExplainer` API interface](../../reference/xwhy/explainers/pix2pix.md)
