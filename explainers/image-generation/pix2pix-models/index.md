# Pix2Pix model examples

Experimental interface — under construction

XWhy currently exports `Pix2PixExplainer`, but its `explain()` method raises `NotImplementedError`. The examples described here are a documentation roadmap, not an executable workflow.

Pix2Pix-style models are one example of conditional image-to-image generation. They belong under **Image Generation → Image Editing** because the model transforms a source image into a target image rather than returning a classification score.

## Planned first example

A future worked example should document:

1. the source image and target transformation task;
1. the Pix2Pix model, weights, and preprocessing;
1. the source-image regions selected for perturbation;
1. the output-distance or perceptual-similarity measure;
1. the local surrogate configuration;
1. the generated attribution map;
1. fidelity, stability, and runtime evidence;
1. limitations of interpreting generative outputs.

## Planned comparison cases

- removing or masking source-image regions;
- perturbing the conditioning input;
- comparing pixel-space and perceptual distances;
- checking attribution stability across generation seeds;
- detecting unintended changes outside the target edit region.

[View the current `Pix2PixExplainer` API interface](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/reference/xwhy/explainers/pix2pix/index.md)
