# Welcome to XWhy

**Explaining black boxes with a SMILE.**

Machine learning is currently undergoing an explosion in capability, popularity, and sophistication. However, one of the major barriers to widespread acceptance of machine learning (ML) models is trustworthiness. Most state-of-the-art ML models operate as complex "black boxes"—their inner workings opaque, mysterious, and difficult to audit.

Explainability is a foundational pillar for establishing this missing trust. **XWhy** was built to bridge this gap. Powered by a novel method called **SMILE** (*Statistical Model-agnostic Interpretability with Local Explanations*), XWhy improves upon previous local explainability frameworks by leveraging robust statistical distance measures. This ensures stable, reliable, and mathematically grounded local explanations applicable across a wide variety of input data domains.

---

## Table of Contents

*   [Home Overview](index.md)
*   [Quick Start & General Usage](#quick-start)
*   [Rich Visualizations](#rich-visualizations)
*   [Advanced Feature: LLM Explainer](llm_explainer.md) — *Learn how to interpret LLM prompt-response dynamics.*

---

## Quick Start

XWhy is entirely model-agnostic, meaning you can hook it up to any model architecture. Here is a high-level conceptual example of how to initialize an explainer and generate feature importance metrics:

```python
import xwhy

# Initialize a standard model-agnostic explainer
explainer = xwhy.Explainer(model=your_blackbox_model, data_domain="text")

# Generate local attribution values for a specific instance
explanation = explainer.explain(instance="Your input data sample here")

# Print structural evaluation metrics
print(explanation.metrics)

```

---

## Rich Visualizations

Providing raw numerical attributions is only half the battle; human interpretability also requires clear and intuitive visual interfaces. XWhy includes a flexible visualization layer for presenting local and global explanation results in formats that are easy to inspect and compare.

After computing statistical attributions through the SMILE engine, XWhy can generate several complementary visualizations, including:

* **Force Plots** to show how individual features push a prediction toward or away from a reference value
* **Waterfall Plots** to trace how feature contributions move the output step by step
* **Summary and Bar Plots** to rank feature importance at local or global level

---

Next Step: If you are looking to explain foundational language models, proceed directly to our [LLM Explainer Guide](llm_explainer.md).
