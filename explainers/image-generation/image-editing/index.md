# Image editing explainability

Coming soon

A supported image-editing explanation workflow is not yet implemented in XWhy.

Image editing differs from image classification because the target model produces a transformed image rather than a class score. An explanation therefore needs to connect changes in the source image, edit instruction, mask, or conditioning input to changes in the generated output.

## Planned questions

Future image-editing explanations should help users examine:

- which source-image regions influenced the edited output;
- which words or phrases in an edit instruction had the strongest effect;
- how an edit mask constrained or redirected generation;
- whether unrelated regions were changed unexpectedly;
- how stable an edit is across seeds or small input changes;
- whether the explanation remains consistent across output-distance measures.

## Planned model families

The section may include examples for:

- Pix2Pix-style conditional image-to-image models;
- instruction-guided image editors;
- diffusion-based image-to-image systems;
- inpainting and outpainting models;
- domain-transfer and style-transfer systems.

Model-specific pages will be added only after the corresponding integration, tests, and reproducible outputs are available.

Next: [Pix2Pix model examples](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/image-generation/pix2pix-models/index.md).
