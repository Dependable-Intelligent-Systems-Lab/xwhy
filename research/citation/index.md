# Cite XWhy and SMILE

Cite the **software artefact** you used and the **research method** that supports your analysis. These are related but distinct citations.

## Citation decision table

| What you used or discussed                                                    | Recommended research citation                              |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------- |
| General SMILE method, local statistical weighting, or classifier explanations | Foundational SMILE paper                                   |
| Point-cloud explanations                                                      | Foundational SMILE paper + point-cloud SMILE preprint      |
| Instruction-based image-editing explanations                                  | Foundational SMILE paper + image-editing SMILE preprint    |
| Large-language-model prompt explanations                                      | Foundational SMILE paper + gSMILE preprint                 |
| Knowledge-graph or GraphRAG explanations                                      | Foundational SMILE paper + KG-SMILE preprint               |
| Concept-based explanation auditing                                            | Foundational SMILE paper + ConceptSMILE preprint           |
| Financial-sentiment reasoning with local perturbation explanations            | gSMILE preprint + financial-sentiment study                |
| The consolidated MSc study of generative-AI explainability                    | MSc thesis; add the relevant method paper where applicable |

## Foundational SMILE citation

Use this as the primary methodological citation when referring to SMILE generally:

> Koorosh Aslansefat, Mojgan Hashemian, Martin Walker, Mohammed Naveed Akram, Ioannis Sorokos, and Yiannis Papadopoulos. “Explaining Black Boxes with a SMILE: Statistical Model-Agnostic Interpretability with Local Explanations.” *IEEE Software*, 41(1), 87–97, 2024. <https://doi.org/10.1109/MS.2023.3321282>

```
@article{aslansefat2024smile,
  author    = {Koorosh Aslansefat and Mojgan Hashemian and Martin Walker
               and Mohammed Naveed Akram and Ioannis Sorokos
               and Yiannis Papadopoulos},
  title     = {Explaining Black Boxes with a {SMILE}: Statistical
               Model-Agnostic Interpretability with Local Explanations},
  journal   = {IEEE Software},
  year      = {2024},
  volume    = {41},
  number    = {1},
  pages     = {87--97},
  month     = {Jan.--Feb.},
  publisher = {IEEE},
  doi       = {10.1109/MS.2023.3321282},
  url       = {https://doi.org/10.1109/MS.2023.3321282}
}
```

## Modality-specific citations

### Point clouds

Cite the foundational SMILE paper and:

> Seyed Mohammad Ahmadi, Koorosh Aslansefat, Rubén Valcarce-Diñeiro, and Joshua Barnfather. “Explainability of Point Cloud Neural Networks Using SMILE: Statistical Model-Agnostic Interpretability with Local Explanations.” arXiv:2410.15374, 2024. <https://arxiv.org/abs/2410.15374>

### Instruction-based image editing

Cite the foundational SMILE paper and:

> Zeinab Dehghani, Koorosh Aslansefat, Adil Khan, Adín Ramírez Rivera, Franky George, and Muhammad Khalid. “Mapping the Mind of an Instruction-Based Image Editing Using SMILE.” arXiv:2412.16277, 2024. <https://arxiv.org/abs/2412.16277>

### Large language models

Cite the foundational SMILE paper and:

> Zeinab Dehghani, Mohammed Naveed Akram, Koorosh Aslansefat, Adil Khan, and Yiannis Papadopoulos. “Explaining Large Language Models with gSMILE.” arXiv:2505.21657, 2025. <https://arxiv.org/abs/2505.21657>

### Knowledge-graph retrieval-augmented generation

Cite the foundational SMILE paper and:

> Zahra Zehtabi Sabeti Moghaddam, Zeinab Dehghani, Maneeha Rani, Koorosh Aslansefat, Bhupesh Kumar Mishra, Rameez Raja Kureshi, and Dhavalkumar Thakker. “Explainable Knowledge Graph Retrieval-Augmented Generation (KG-RAG) with KG-SMILE.” arXiv:2509.03626, 2025. <https://arxiv.org/abs/2509.03626>

### Concept-based explainability

Cite the foundational SMILE paper and:

> Mohadeseh Mollapour, Koorosh Aslansefat, Zeinab Dehghani, Bhupesh Kumar Mishra, Tejal Shah, and Zhibao Mian. “ConceptSMILE: Auditing the Trustworthiness of Concept-Based Explainable AI.” arXiv:2607.09649, 2026. <https://arxiv.org/abs/2607.09649>

### Financial-sentiment reasoning

For the applied financial-sentiment study, cite gSMILE and:

> Sania Verma, Koorosh Aslansefat, Joyjit Chatterjee, Akash Marar, Anu Mehra, and Aisha Ekundayo. “When Words Move Markets: Interpretable Behavioural and Robustness Analysis of LLMs for Financial Sentiment Reasoning via Local Perturbation Explanations.” In *Natural Language Processing and Information Systems: NLDB 2026*, LNCS 16696, 271–286. Springer, 2027; first published online 4 July 2026. <https://doi.org/10.1007/978-3-032-29532-3_19>

The [publication catalogue](https://dependable-intelligent-systems-lab.github.io/xwhy/research/publications/index.md) provides complete BibTeX records for every method and study.

## Citing the XWhy software

A research-method citation does not identify the exact software revision used in an experiment. Until the repository provides a versioned archived software release and formal software DOI, record:

- the package version;
- the Git commit SHA when reproducibility is important;
- the repository URL;
- the date on which the software was accessed;
- the configuration, model, data, and random seed used.

A temporary plain-text software reference can use this form:

> XWhy contributors. *XWhy: Model-Agnostic Explainability with SMILE*. Version or commit used. GitHub repository: <https://github.com/Dependable-Intelligent-Systems-Lab/xwhy>. Accessed YYYY-MM-DD.

## Citation principles

1. **Cite the foundational method** when the work depends on SMILE’s statistical local-surrogate formulation.
1. **Add the relevant extension** when the explained system is a point cloud, image editor, LLM, KG-RAG pipeline, or concept-based model.
1. **Distinguish preprints from peer-reviewed publications** in prose and bibliographies.
1. **List the thesis separately** from journal and conference papers.
1. **Record the software revision** independently from the paper citation so an experiment can be reproduced.
