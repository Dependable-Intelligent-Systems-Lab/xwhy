---
title: Image Generation and Editing Explainability with XWhy
description: Use XWhy to explain image-generation and image-editing systems through prompt perturbation, output-distance analysis, and local surrogate modelling.
---

# Image generation and editing explainability

!!! success "Available"
    `ImageGenerationAndEditingExplainer` is implemented and exported by XWhy for supported image-generation and image-editing providers, pipelines, and compatible custom generation functions.

Image-generation and image-editing models produce an image rather than a class score. XWhy therefore explains their behaviour by perturbing the textual instruction, generating or editing images for those perturbations, measuring how the visual outputs change, and fitting a local surrogate model.

## Basic use

```python
from xwhy import ImageGenerationAndEditingExplainer

explainer = ImageGenerationAndEditingExplainer(
    engine="openai",
    model_name="dall-e-3",
    num_perturbations=64,
)

result = explainer.explain(
    "A red bicycle beside a stone wall."
)
```

Provider credentials and model-specific arguments must be configured for the selected generation service.

For image editing, provide a source image:

```python
result = explainer.explain(
    "Change the bicycle from red to blue.",
    input_image_path="bicycle.png",
)
```

## Current explanation workflow

The implemented workflow can:

1. generate text perturbations around the original instruction;
2. generate a reference image or reference edit;
3. generate or edit images for the perturbed instructions;
4. compare each perturbed output with the reference output using a configurable image-distance measure;
5. compute semantic distance between the original and perturbed text;
6. use those distances to weight the local neighbourhood;
7. fit a local surrogate model; and
8. return word-level coefficients together with surrogate-quality metrics and raw evidence.

## Model integration

`ImageGenerationAndEditingExplainer` can work with supported providers and can also be configured with a compatible pipeline, custom model, or custom generation function. The public component is deliberately broader than any one model family.

Pix2Pix is therefore documented as one conditional image-to-image model family within this capability, not as the name of the XWhy explainer. See [Pix2Pix-style models](pix2pix-models.md).

## What the explanation means

The coefficients estimate how perturbations to words in the conditioning instruction are locally associated with changes in the generated or edited image under the selected output-distance measure. They do not expose the model's hidden reasoning and should not be interpreted as causal proof.

Generative models can also be stochastic. For research use, report the model/provider, seed where supported, perturbation count, text and image distance settings, surrogate configuration, and fidelity metrics.

## Subsections

- [Image editing](image-editing.md)
- [Pix2Pix-style model examples](pix2pix-models.md)

[View the current image explainer API reference](../../reference/xwhy/explainers/image.md)
