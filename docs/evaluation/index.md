---
title: Evaluate XWhy Explanations
description: Current status and planned guidance for assessing fidelity, stability, robustness, runtime, and usefulness of XWhy explanations.
---

# Evaluate explanations

Implemented XWhy result objects expose evaluation metrics, and current tutorials show how to inspect `result.metrics`.

!!! warning "Metric catalogue under construction"
    A verified catalogue defining every metric, its direction, range, assumptions, and recommended interpretation is still being prepared. Do not invent universal acceptance thresholds.

The evaluation documentation will cover:

- local surrogate fidelity;
- stability across seeds and nearby inputs;
- perturbation realism;
- sensitivity to distance and surrogate choices;
- runtime and provider cost;
- comparison with baselines;
- task-specific human usefulness.

Until that catalogue is complete, report metric names, configurations, raw values, and comparison conditions transparently.
