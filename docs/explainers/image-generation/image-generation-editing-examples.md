---
title: Image Generation and Editing Examples
description: Comprehensive examples using XWhy's ImageGenerationAndEditingExplainer to explain generative models across OpenAI, ByteDance, Gemini, HuggingFace, and custom APIs.
---

# Image generation and editing examples

The `ImageGenerationAndEditingExplainer` provides a model-agnostic workflow for understanding how text prompts and edit instructions influence generated or edited images. This capability applies to various providers (OpenAI, Gemini, ByteDance) and custom pipelines.

!!! important "API Compatibility Note"
    Except for Google (Gemini) and HuggingFace, XWhy leverages an OpenAI-compatible interface for all supported providers, including ByteDance. When using these providers, refer to their respective image generation and editing API documentation to understand the parameters to send. For example, for ByteDance, consult [their API documentation](https://docs.byteplus.com/en/docs/ModelArk/1824121#9695d195).

## Explanation Scenarios

For generative image models, the workflow is typically divided into two parts:

1. **Generation**: Requires only a text prompt.
2. **Editing**: Requires both an edit instruction (prompt) and a source image (`input_image_path`).

Below are examples of how to configure the explainer for several supported backends.

---

## 1. OpenAI Provider

```python
import xwhy
from xwhy import ImageGenerationAndEditingExplainer

# For Generation
explainer_gen = ImageGenerationAndEditingExplainer(
    engine="openai",
    model_name="gpt-image-2",
    use_image_embedding_model=True,
    num_perturbations=5,
)

result_gen = explainer_gen.explain(
    instance="A futuristic train bursting from a black hole, cinematic lighting",
    normalization_method="inverse",
    quality="low",
    size="1024x1024",
    n=1,
)

# For Editing
explainer_edit = ImageGenerationAndEditingExplainer(
    engine="openai",
    model_name="gpt-image-2",
    use_image_embedding_model=True,
    num_perturbations=5,
)

result_edit = explainer_edit.explain(
    instance="transform it to become blur",
    input_image_path="./cat-and-dog.jpg",
    normalization_method="inverse",
    quality="low",
    size="1024x1024",
    n=1,
)
```

---

## 2. ByteDance Provider

```python
# For Generation
explainer_bd = ImageGenerationAndEditingExplainer(
    engine="bytedance",
    model_name="seedream-5-0-260128",
    use_image_embedding_model=True,
    num_perturbations=5,
)

result_bd = explainer_bd.explain(
    instance="A futuristic train bursting from a black hole, cinematic lighting",
    normalization_method="inverse",
    size="2K", # ByteDance images must be at least 3686400 pixels
    output_format="jpeg",
    response_format="url",
    extra_body={"watermark": False},
)
```

---

## 3. Gemini Provider

```python
# For Generation (Batch mode enabled)
explainer_gemini = ImageGenerationAndEditingExplainer(
    engine="gemini",
    model_name="gemini-2.5-flash-image",
    use_image_embedding_model=True,
    num_perturbations=5,
    temperature=1.0,
)

result_gemini = explainer_gemini.explain(
    instance="A futuristic train bursting from a black hole, cinematic lighting",
    normalization_method="inverse",
    batch=True,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
)
```

---

## 4. HuggingFace Provider

### Instruct-Pix2Pix Editing

```python
explainer_hf = ImageGenerationAndEditingExplainer(
    engine="huggingface",
    model_name="timbrooks/instruct-pix2pix",
    use_image_embedding_model=True,
    num_perturbations=5,
    use_segmentation_model=True,
)

result_hf = explainer_hf.explain(
    instance="transform it to become blur",
    input_image_path="./cat-and-dog.jpg",
    normalization_method="inverse",
    num_inference_steps=10,
)
```

---

## Visualizing Results

Once an explanation is generated, you can visualize the estimated term contributions using the standard result plots:

```python
# General plots
result_gen.plot()
xwhy.plots.text_heatmap(result_gen)
xwhy.plots.plot_feature_bar_chart(result_gen)
xwhy.plots.plot_feature_box_plot(result_gen)

# SHAP-style attribution plots
xwhy.plots.bar(result_gen)
xwhy.plots.waterfall(result_gen)
xwhy.plots.text(result_gen)
xwhy.plots.force(result_gen)
xwhy.plots.decision(result_gen)
```

[Read the image generation and editing overview](index.md)
