# Limitations and responsible use

XWhy explanations are evidence about sampled black-box behaviour, not direct access to a model's internal reasoning.

## Main limitations

- **Approximation:** the surrogate can disagree with the target model.
- **Local scope:** conclusions may not generalise beyond the sampled neighbourhood.
- **Perturbation dependence:** unrealistic perturbations can create misleading attributions.
- **Metric dependence:** one quality metric cannot establish explanation validity.
- **Non-causality:** feature importance does not prove causal influence.
- **Instability:** explanations may change with seeds, inputs, model versions, or provider behaviour.
- **Data and privacy:** prompts, images, and outputs may contain sensitive information.

## High-impact use

Do not use an XWhy plot as the sole basis for medical, legal, financial, employment, safety, or regulatory decisions. Combine explanation evidence with domain review, model validation, uncertainty analysis, robustness testing, and documented human oversight.
