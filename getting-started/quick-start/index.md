# Quick start

XWhy currently provides two executable explanation workflows.

## Explain an image classifier

Use `ImageClassificationExplainer` for a PyTorch image-classification model. The explainer estimates how image regions influence a selected prediction.

[Open the image-classification tutorial](https://dependable-intelligent-systems-lab.github.io/xwhy/image_classification_explainer/index.md)

## Explain an LLM response

Use `LLMExplainer` to perturb a text prompt, compare the resulting model responses, and estimate local token or word influence.

[Open the LLM tutorial](https://dependable-intelligent-systems-lab.github.io/xwhy/llm_explainer/index.md)

## Do not use development interfaces as implemented explainers

`TabularExplainer`, `TextExplainer`, and `PointCloudExplainer` are currently public development interfaces whose `explain()` methods are not implemented.

Image-generation explainability is also under construction. The current `Pix2PixExplainer` class is an early interface within the broader image-generation and image-editing roadmap; it is not yet an executable workflow.

See the [explainer status matrix](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/index.md) before designing a workflow.
