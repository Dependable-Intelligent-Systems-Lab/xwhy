---
title: Does the Explanation Hold Up?
description: Practical checks for the fit, stability and faithfulness of a local explanation.
---

[← All research and insights](../blogs.md){ .xwhy-blog-back }

# Does the explanation hold up?

![An original illustration of an explanation passing several evaluation checks](../../../assets/images/blogs/explanation-fidelity.svg)

*Evaluation · XWhy explainer*

An explanation can be easy to read and still be misleading. A highlighted word or image region should invite a further test: does the model behave as the explanation suggests when that part of the input changes?

## Three checks answer different questions

**Local fit or fidelity** asks whether the simpler explanation model approximates the target model on the sampled inputs around the example. A poor fit weakens confidence in the reported contributions even when the chart looks convincing. A good fit is useful, but it remains limited to the neighbourhood that was sampled.

**Stability** asks whether the explanation remains similar after a small change that should not change its meaning. If a harmless rephrasing moves all the important terms, investigate the perturbation method, sampling variation and the model's own sensitivity.

**Intervention checks** compare a stated attribution with the model's observed response after a selected feature is removed, replaced or preserved. If a feature claimed to dominate has little measured effect under relevant interventions, the claim needs revision. Feature changes must remain meaningful; replacing an image patch with an unnatural artefact can create a different problem.

## Report the evidence, not only the visual

An evaluation should identify the target model and input, the output being measured, the perturbation rules, the number of samples, and the score or comparison used. Repeat runs where randomness matters. When possible, show both the explanation and a small set of actual model outputs that support or challenge it.

The [XWhy evaluation overview](../../../evaluation/index.md), [fidelity guidance](../../../evaluation/attribution-fidelity.md) and [faithfulness guidance](../../../evaluation/attribution-faithfulness.md) provide more specific measures. No one measure proves that an explanation is complete or that the underlying model is safe.
