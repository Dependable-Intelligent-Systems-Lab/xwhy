---
title: Make XWhy Explanations Reproducible
description: Record model, data, random seed, perturbation, provider, and software information needed to reproduce an XWhy explanation.
---

# Reproducible explanations

Record the following with every experiment:

- XWhy and Python versions;
- model name, version, weights, and preprocessing;
- exact input or dataset identifier;
- random seed;
- number and type of perturbations;
- embedding and distance settings;
- surrogate model and selection mode;
- provider and model parameters for LLM experiments;
- hardware and execution date;
- returned metrics and saved plots.

For remote LLMs, identical settings may still produce different outputs because providers can change model versions or serving behaviour. Store representative raw responses when licences and privacy constraints permit.
