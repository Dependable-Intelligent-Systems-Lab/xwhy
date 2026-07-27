---
title: Understand XWhy Explanation Results
description: Understand the structured outputs, feature contributions, metrics, and interpretation limits of XWhy explanation results.
---

# Explanation results

Implemented explainers return a structured result object containing explanation outputs and evaluation information.

Typical result content includes:

- local feature or region contributions;
- top-ranked influential components;
- the model output being explained;
- explanation-quality metrics;
- data required by supported plots.

The exact fields depend on the modality. Consult the generated [API reference](../reference/) for the authoritative Python interface.

!!! warning "Interpretation boundary"
    A high-ranked feature is associated with changes in the local surrogate approximation. It is not automatically a causal factor, a globally important feature, or evidence of the model's internal reasoning process.

The metric catalogue and recommended acceptance thresholds are [under construction](../evaluation/index.md).
