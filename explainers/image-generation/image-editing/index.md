# Image editing explainability

Available

Image editing is supported through `ImageGenerationAndEditingExplainer` when the selected provider, pipeline, or custom engine supports editing.

Image editing differs from image classification because the model produces a transformed image rather than a class score. XWhy therefore evaluates how changes to the edit instruction affect the resulting image relative to a reference edit.

## Basic use

```
from xwhy import ImageGenerationAndEditingExplainer

explainer = ImageGenerationAndEditingExplainer(
    engine="openai",
    model_name="gpt-image-1",
    num_perturbations=64,
)

result = explainer.explain(
    "Replace the cloudy sky with a clear blue sky.",
    input_image_path="scene.png",
)
```

The exact provider and model must support image editing, and provider-specific credentials or arguments may be required.

## What XWhy perturbs

The current implementation focuses on perturbing the **textual editing instruction**. For each perturbation it reruns the editing model, compares the perturbed edited output with the reference edited output, and fits a local surrogate model.

This allows questions such as:

- Which words in the edit instruction had the strongest measured influence on the edited output?
- Does removing a key object, attribute, or action word substantially change the result?
- Is the explanation stable when the edit is repeated under controlled settings?
- Does the local surrogate provide an adequate approximation of the observed edits?

## Current scope and limitations

The present implementation should not be described as direct pixel-level causal attribution to the source image or mask. Its principal explanation units are terms in the textual instruction, with output changes measured in image space or an image-embedding space depending on configuration.

Future extensions can add explicit source-region, mask-region, and cross-modal attribution while retaining the same intervention-based evaluation principles.

Because image-generation and editing models may be stochastic, explanation results can vary across executions. Where the provider supports it, control and report the seed, model version, perturbation count, output-distance measure, and surrogate fidelity.

See the [image generation and editing overview](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/image-generation/index.md) and [Pix2Pix-style model examples](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/image-generation/pix2pix-models/index.md).

[View the current image explainer API reference](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/xwhy/explainers/image/index.md)
