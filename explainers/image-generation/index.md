# Image generation explainability

Under construction

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

- [Image editing](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/image-generation/image-editing/index.md)
- [Pix2Pix model examples](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/image-generation/pix2pix-models/index.md)

## Current implementation status

The current code contains `Pix2PixExplainer` as an early interface. It should remain described as an implementation prototype until its behaviour, accepted inputs, perturbation strategy, result type, tests, and examples are complete.

[View the current Pix2Pix API interface](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/xwhy/explainers/pix2pix/index.md)
