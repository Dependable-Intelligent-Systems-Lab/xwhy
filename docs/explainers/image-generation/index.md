---
title: Image Generation Explainability with XWhy
description: Planned XWhy support for explaining image-generation and image-editing systems, including diffusion, instruction-guided editing, and Pix2Pix-style models.
---

# Image generation explainability

!!! warning "Under construction"
    Image-generation explainability is not yet available as a supported XWhy workflow. The repository currently exposes an early `Pix2PixExplainer` interface, but its `explain()` method is not implemented.

This section is the public entry point for explainability methods applied to systems that create or transform images. **Pix2Pix is treated as one model-family example within this broader capability, not as the name of the capability itself.**

## Planned coverage

Image-generation documentation is expected to cover:

- text-to-image generation;
- image-to-image generation;
- instruction-guided image editing;
- inpainting and region replacement;
- conditional generation;
- attribution to prompts, source-image regions, masks, and conditioning inputs;
- output similarity and perceptual-distance measures;
- explanation stability, faithfulness, and safety limitations.

## Subsections

- [Image editing](image-editing.md)
- [Pix2Pix model examples](pix2pix-models.md)

## Current implementation status

The current code contains `Pix2PixExplainer` as an early interface. It should remain described as an implementation prototype until its behaviour, accepted inputs, perturbation strategy, result type, tests, and examples are complete.

[View the current Pix2Pix API interface](../../reference/xwhy/explainers/pix2pix.md)
