# XWhy: eXplain Why

**Explain black-box behaviour with a SMILE.**

**XWhy: eXplain Why** is a Python library for model-agnostic local explainability. It uses **SMILE**—Statistical Model-agnostic Interpretability with Local Explanations—to perturb an input, observe changes in model behaviour, fit a local surrogate model, and report feature-level influence together with explanation-quality evidence.

Current package maturity

XWhy `v0.0.3` is currently classified as **pre-alpha**. Tabular, Image and LLM explainers are implemented. Other capabilities are clearly labelled as under construction or coming soon.

## Start here

- [Install XWhy](https://dependable-intelligent-systems-lab.github.io/xwhy/getting-started/installation/index.md)
- [Generate your first explanation](https://dependable-intelligent-systems-lab.github.io/xwhy/getting-started/quick-start/index.md)
- [Choose the correct explainer](https://dependable-intelligent-systems-lab.github.io/xwhy/getting-started/choosing-an-explainer/index.md)
- [Browse all explainers and their status](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/index.md)
- [Read the generated API reference](https://dependable-intelligent-systems-lab.github.io/xwhy/reference/index.md)

## Capability overview

| Capability                   | Public component               | Documentation status | Implementation status |
| ---------------------------- | ------------------------------ | -------------------- | --------------------- |
| Image classification         | `ImageClassificationExplainer` | Available            | Available             |
| Image generation and editing | `Pix2PixExplainer` prototype   | Under construction   | Interface only        |
| LLM prompt-response          | `LLMExplainer`                 | Available            | Available             |
| Tabular                      | `TabularExplainer`             | Under construction   | Interface only        |
| Text                         | `TextExplainer`                | Under construction   | Interface only        |
| Point cloud                  | `PointCloudExplainer`          | Under construction   | Interface only        |
| Time series                  | Planned                        | Coming soon          | Not yet implemented   |
| Multimodal                   | Planned                        | Coming soon          | Not yet implemented   |

Pix2Pix is documented as one image-editing model family within the broader [image-generation explainability](https://dependable-intelligent-systems-lab.github.io/xwhy/explainers/image-generation/index.md) section.

## What XWhy explanations mean

XWhy produces local, perturbation-based approximations of model behaviour around a selected input. An explanation can identify associations between input components and changes in model output, but it does not expose a model's private internal reasoning or prove causality.

Read [limitations and responsible use](https://dependable-intelligent-systems-lab.github.io/xwhy/concepts/limitations/index.md) before using explanations in safety-critical, medical, legal, financial, or high-impact decisions.
