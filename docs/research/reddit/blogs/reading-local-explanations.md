---
title: What a Local Explanation Tells You
description: How to read a single-input explanation without treating it as a claim about the whole model.
---

[← All research and insights](../blogs.md){ .xwhy-blog-back }

# What a local explanation tells you

![An original illustration of an image and its local attribution map](../../../assets/images/blogs/local-explanations.svg)

*Explainability · XWhy explainer*

A local explanation describes how a model behaved around a *particular* input under a chosen set of changes. If an image classifier responds strongly when the dog's face is hidden, that observation can help explain this prediction. It cannot tell you that the face is always the decisive feature across the entire dataset.

## Read the result together with its question

Before interpreting a chart, identify the selected input, the model output being explained and what counts as a changed feature. For language models, that might mean removing prompt words. For an image model, it might mean changing image regions. The explanation should also tell you what comparison and local model produced the contributions.

A large positive contribution often means a feature was associated with a higher local score under those settings. A small value could mean the feature was unimportant nearby, or that the chosen perturbations did not isolate its influence well. The direction and units of a score depend on the output and explanation method; do not assume every heatmap or contribution plot uses the same scale.

## Questions worth asking

- Does the simpler local model reproduce the observed behaviour of the model being explained?
- Do small, harmless changes to the input leave the main finding intact?
- Would another sensible way of changing the input produce a similar conclusion?
- Are the highlighted regions or words relevant to the actual task?

An explanation is a tool for examining behaviour, not proof of an internal reasoning process. It also does not establish whether a prediction is fair, safe or correct. For high-impact uses, combine local explanations with broader performance tests, subgroup checks and expert review.

Read the [local explanations guide](../../../concepts/local-explanations.md) and [limitations](../../../concepts/limitations.md) for more detail.
