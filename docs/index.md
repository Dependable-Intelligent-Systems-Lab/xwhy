---
title: "XWhy: eXplain Why"
description: Use XWhy and SMILE to explain image classification, image generation and editing, LLM, tabular, and text models, with a roadmap for point-cloud, time-series, multimodal, agentic AI, and multi-agent AI explainability.
---

# XWhy: eXplain Why

**Explain black-box behaviour with a SMILE.**

**XWhy: eXplain Why** is a Python library for model-agnostic local explainability. It uses **SMILE**—Statistical Model-agnostic Interpretability with Local Explanations—to perturb an input, observe changes in model behaviour, fit a local surrogate model, and report feature-level influence together with explanation-quality evidence.

!!! info "Current package maturity"
    XWhy `v{{ XWHY_VERSION }}` is currently classified as **pre-alpha**. Image Classification, Image Generation & Editing, LLM, Tabular, and Text explainers are implemented. Point Cloud remains a development interface, while Time Series, Multimodal, Agentic AI, and Multi-Agent AI are planned capabilities.

## Start here

- [Install XWhy](getting-started/installation.md)
- [Generate your first explanation](getting-started/quick-start.md)
- [Choose the correct explainer](getting-started/choosing-an-explainer.md)
- [Browse all explainers and their status](explainers/index.md)
- [Read the generated API reference](reference/index.md)

## Capability overview

| Capability | Public component | Explanation focus | Documentation status | Implementation status |
| --- | --- | --- | --- | --- |
| Image classification | `ImageClassificationExplainer` | Local image-region influence on class prediction | Available | Available |
| Image generation and editing | `ImageGenerationAndEditingExplainer` | Influence of prompt terms and conditioning inputs on generated or edited images | Available | Available |
| LLM prompt-response | `LLMExplainer` | Local influence of prompt terms or phrases on response behaviour | Available | Available |
| Tabular | `TabularExplainer` | Influence of structured features on classification or regression | Available | Available |
| Text | `TextExplainer` | Word-level influence on conventional text-model predictions | Available | Available |
| Point cloud | `PointCloudExplainer` | Influence of 3D points or point groups on model prediction | Under construction | Interface only |
| Time series | Planned | Influence of observations, windows, and temporal patterns on predictions | Coming soon | Not yet implemented |
| Multimodal | Planned | Modality-specific and cross-modal contributions and interactions | Coming soon | Not yet implemented |
| Agentic AI | Planned | Plans, retrieval, memory, tool use, actions, state transitions, and uncertainty or failure propagation within an autonomous agent | Coming soon | Not yet implemented |
| Multi-Agent AI | Planned | Agent contributions, inter-agent messages, coordination, disagreement, dependencies, and uncertainty or failure propagation across agents | Coming soon | Not yet implemented |

The current generative-image component is `ImageGenerationAndEditingExplainer`. Pix2Pix is retained in the documentation as one conditional image-to-image model family, not as the name of the public XWhy explainer.

## Explainability of Agentic AI

Agentic AI changes the explanation target from a single prediction or response to a **sequence of decisions and actions**. An agent may plan, retrieve information, use memory, call tools, evaluate intermediate results, revise its plan, and then act. XWhy's research roadmap therefore extends local explainability from model outputs to the observable behaviour of an agentic workflow.

Planned Agentic AI explainability will address questions such as:

- Why was a particular action, tool, retrieval result, or plan step selected?
- Which observable inputs, retrieved evidence, memory items, or intermediate states had the strongest influence on the final behaviour?
- Where did uncertainty or failure first arise, and how did it propagate through the agent workflow?
- How would the outcome change if a plan step, tool output, memory item, or retrieved item were perturbed or removed?
- Is an explanation stable across repeated executions and small changes to the agent state?

The goal is to explain **observable agent behaviour and intervention effects**, not to claim access to hidden chain-of-thought or private internal reasoning. See the [Agentic AI explainability roadmap](explainers/agentic-ai.md).

## Explainability of Multi-Agent AI

Multi-Agent AI introduces an additional system-level problem: an outcome may emerge from the interaction of several agents rather than from one model or one agent alone. A useful explanation must therefore describe both **what each agent contributed** and **how interactions between agents changed the system outcome**.

Planned Multi-Agent AI explainability will focus on:

- agent-level contribution and responsibility for a system outcome;
- influence of inter-agent messages and shared information;
- coordination, dependency, delegation, and disagreement between agents;
- propagation of uncertainty, errors, or failures from one agent to another;
- counterfactual agent or message ablation to test whether an agent or interaction was necessary for the observed behaviour;
- local agent explanations alongside a system-level explanation of the complete multi-agent trajectory.

This direction is intended to support more transparent analysis of collaborative, competitive, and hierarchical multi-agent systems while keeping explanation claims tied to observable evidence and controlled interventions. See the [Multi-Agent AI explainability roadmap](explainers/multi-agent-ai.md).

## What XWhy explanations mean

XWhy produces local, perturbation-based approximations of model behaviour around a selected input. An explanation can identify associations between input components and changes in model output, but it does not expose a model's private internal reasoning or prove causality.

Read [limitations and responsible use](concepts/limitations.md) before using explanations in safety-critical, medical, legal, financial, or high-impact decisions.
