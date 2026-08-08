# How SMILE works

SMILE means **Statistical Model-agnostic Interpretability with Local Explanations**.

At a high level, an implemented XWhy pipeline:

1. selects the input instance to explain;
1. creates local perturbations of meaningful input components;
1. queries the target black-box model or provider;
1. measures how outputs change using a suitable distance or similarity method;
1. weights observations according to local relevance;
1. fits or selects a surrogate model around the selected instance;
1. reports estimated local contributions and evaluation metrics.

## Why statistical distances matter

Different modalities require different definitions of change. A useful distance measure must reflect meaningful changes in model behaviour rather than only low-level input differences.

The exact distance, perturbation, and surrogate choices are modality-specific. Consult the current explainer tutorial and API reference rather than assuming that settings transfer unchanged across image and language tasks.
