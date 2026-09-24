# Local explanations

A local explanation approximates model behaviour around one selected input and a defined perturbation neighbourhood.

It can answer questions such as:

- Which input components were associated with the largest local output changes?
- Did those components increase or decrease the surrogate output?
- How well did the surrogate approximate sampled model behaviour?

It does not automatically answer:

- Which features are globally important across the dataset?
- What causal mechanism produced the prediction?
- What internal reasoning process the model used?
- Whether the prediction or explanation is fair, safe, or correct?

Local explanations should be compared across nearby inputs, seeds, perturbation settings, and relevant subgroups before strong conclusions are drawn.
