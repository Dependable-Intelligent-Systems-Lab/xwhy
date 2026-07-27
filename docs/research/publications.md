---
title: XWhy and SMILE Publications
description: Complete publication catalogue for the SMILE explainability family, including DOI and arXiv links, publication status, method scope, and BibTeX records.
---

# Publications

This catalogue connects the SMILE research family to the systems each publication explains. The foundational paper defines the core statistical local-explanation method; later publications extend or apply that principle to point clouds, image editing, large language models, knowledge-graph retrieval, concepts, and financial sentiment reasoning.

## Publication catalogue

| Year | Method or study | Explained system | Type | Persistent identifier |
| --- | --- | --- | --- | --- |
| 2024 | Foundational SMILE | ML and deep-learning classifiers | IEEE Software article | [DOI](https://doi.org/10.1109/MS.2023.3321282) |
| 2024 | Point-cloud SMILE | Point-cloud neural networks | arXiv preprint | [arXiv:2410.15374](https://arxiv.org/abs/2410.15374) |
| 2024 | Image-editing SMILE | Instruction-based image editing | arXiv preprint | [arXiv:2412.16277](https://arxiv.org/abs/2412.16277) |
| 2025 | gSMILE | Large language models | arXiv preprint | [arXiv:2505.21657](https://arxiv.org/abs/2505.21657) |
| 2025 | KG-SMILE | Knowledge-graph and GraphRAG systems | arXiv preprint | [arXiv:2509.03626](https://arxiv.org/abs/2509.03626) |
| 2026 | ConceptSMILE | Concept-based explainable AI | arXiv preprint | [arXiv:2607.09649](https://arxiv.org/abs/2607.09649) |
| 2026/2027 | Financial-sentiment local perturbation study | LLM financial-sentiment reasoning | NLDB 2026 conference chapter; 2027 volume | [DOI](https://doi.org/10.1007/978-3-032-29532-3_19) |
| 2025 | Generative-AI SMILE thesis | LLM and image-editing explainability | MSc thesis | [arXiv:2602.01206](https://arxiv.org/abs/2602.01206) |

## Core method

### Explaining Black Boxes with a SMILE

**Koorosh Aslansefat, Mojgan Hashemian, Martin Walker, Mohammed Naveed Akram, Ioannis Sorokos, and Yiannis Papadopoulos.** “Explaining Black Boxes with a SMILE: Statistical Model-Agnostic Interpretability with Local Explanations.” *IEEE Software*, 41(1), 87–97, 2024.

The foundational paper introduces model-agnostic local explanations for black-box machine-learning and deep-learning classifiers. SMILE uses statistical distance measures to weight perturbed samples when fitting a local surrogate model, rather than relying only on geometric proximity.

[DOI](https://doi.org/10.1109/MS.2023.3321282) · [IEEE Xplore](https://ieeexplore.ieee.org/document/10269706)

??? note "BibTeX"

    ```bibtex
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

## Vision and spatial AI

### Explainability of Point Cloud Neural Networks Using SMILE

**Seyed Mohammad Ahmadi, Koorosh Aslansefat, Rubén Valcarce-Diñeiro, and Joshua Barnfather.** “Explainability of Point Cloud Neural Networks Using SMILE: Statistical Model-Agnostic Interpretability with Local Explanations.” arXiv:2410.15374, 2024.

This extension identifies local groups or clusters of 3D points that most strongly influence a point-cloud neural network. Its evaluation considers explanation fidelity, stability, and robustness, with relevance to robotics, LiDAR perception, autonomous systems, and industrial inspection.

[Abstract](https://arxiv.org/abs/2410.15374) · [PDF](https://arxiv.org/pdf/2410.15374)

??? note "BibTeX"

    ```bibtex
    @misc{ahmadi2024pointcloudsmile,
      author        = {Seyed Mohammad Ahmadi and Koorosh Aslansefat
                       and Rub{\'e}n Valcarce-Di{\~n}eiro
                       and Joshua Barnfather},
      title         = {Explainability of Point Cloud Neural Networks Using
                       {SMILE}: Statistical Model-Agnostic Interpretability
                       with Local Explanations},
      year          = {2024},
      eprint        = {2410.15374},
      archivePrefix = {arXiv},
      primaryClass  = {cs.LG},
      doi           = {10.48550/arXiv.2410.15374},
      url           = {https://arxiv.org/abs/2410.15374}
    }
    ```

### Mapping the Mind of an Instruction-Based Image Editing Using SMILE

**Zeinab Dehghani, Koorosh Aslansefat, Adil Khan, Adín Ramírez Rivera, Franky George, and Muhammad Khalid.** “Mapping the Mind of an Instruction-Based Image Editing Using SMILE.” arXiv:2412.16277, 2024.

This work applies SMILE to instruction-based generative image-editing systems. It perturbs words or phrases in the editing instruction and measures changes in the generated image, revealing which parts of the instruction drive particular visual modifications.

[Abstract](https://arxiv.org/abs/2412.16277) · [PDF](https://arxiv.org/pdf/2412.16277)

??? note "BibTeX"

    ```bibtex
    @misc{dehghani2024imageeditingsmile,
      author        = {Zeinab Dehghani and Koorosh Aslansefat
                       and Adil Khan and Ad{\'i}n Ram{\'i}rez Rivera
                       and Franky George and Muhammad Khalid},
      title         = {Mapping the Mind of an Instruction-Based Image
                       Editing Using {SMILE}},
      year          = {2024},
      eprint        = {2412.16277},
      archivePrefix = {arXiv},
      primaryClass  = {cs.AI},
      doi           = {10.48550/arXiv.2412.16277},
      url           = {https://arxiv.org/abs/2412.16277}
    }
    ```

## Language and generative AI

### Explaining Large Language Models with gSMILE

**Zeinab Dehghani, Mohammed Naveed Akram, Koorosh Aslansefat, Adil Khan, and Yiannis Papadopoulos.** “Explaining Large Language Models with gSMILE.” arXiv:2505.21657, 2025.

gSMILE perturbs prompt tokens or textual components, observes changes in model outputs, measures those changes with statistical distances such as Wasserstein distance, and fits a locally weighted surrogate model. The resulting token-level explanations can be used to study prompt influence, response sensitivity, instability, and potentially unsafe dependence on particular prompt elements.

[Abstract](https://arxiv.org/abs/2505.21657) · [PDF](https://arxiv.org/pdf/2505.21657)

??? note "BibTeX"

    ```bibtex
    @misc{dehghani2025gsmile,
      author        = {Zeinab Dehghani and Mohammed Naveed Akram
                       and Koorosh Aslansefat and Adil Khan
                       and Yiannis Papadopoulos},
      title         = {Explaining Large Language Models with {gSMILE}},
      year          = {2025},
      eprint        = {2505.21657},
      archivePrefix = {arXiv},
      primaryClass  = {cs.CL},
      doi           = {10.48550/arXiv.2505.21657},
      url           = {https://arxiv.org/abs/2505.21657}
    }
    ```

### When Words Move Markets

**Sania Verma, Koorosh Aslansefat, Joyjit Chatterjee, Akash Marar, Anu Mehra, and Aisha Ekundayo.** “When Words Move Markets: Interpretable Behavioural and Robustness Analysis of LLMs for Financial Sentiment Reasoning via Local Perturbation Explanations.” In *Natural Language Processing and Information Systems: NLDB 2026*, LNCS 16696, 271–286. Springer, 2027; first published online 4 July 2026.

This study applies local perturbation explanations derived from the gSMILE approach to financial-sentiment reasoning. It investigates which words, phrases, and contextual cues influence an LLM’s financial-sentiment outcome and whether the behaviour and explanation remain stable under rewording or small perturbations.

[DOI](https://doi.org/10.1007/978-3-032-29532-3_19) · [SpringerLink](https://link.springer.com/chapter/10.1007/978-3-032-29532-3_19)

??? note "BibTeX"

    ```bibtex
    @inproceedings{verma2027wordsmarkets,
      author    = {Sania Verma and Koorosh Aslansefat and Joyjit Chatterjee
                   and Akash Marar and Anu Mehra and Aisha Ekundayo},
      title     = {When Words Move Markets: Interpretable Behavioural and
                   Robustness Analysis of {LLM}s for Financial Sentiment
                   Reasoning via Local Perturbation Explanations},
      booktitle = {Natural Language Processing and Information Systems:
                   {NLDB} 2026},
      editor    = {Elena Cabrio and Eric Monteiro},
      series    = {Lecture Notes in Computer Science},
      volume    = {16696},
      pages     = {271--286},
      year      = {2027},
      publisher = {Springer},
      address   = {Cham},
      doi       = {10.1007/978-3-032-29532-3_19},
      isbn      = {978-3-032-29532-3},
      url       = {https://doi.org/10.1007/978-3-032-29532-3_19},
      note      = {First published online 4 July 2026;
                   presented at NLDB 2026}
    }
    ```

## Retrieval-augmented generation

### Explainable Knowledge Graph Retrieval-Augmented Generation with KG-SMILE

**Zahra Zehtabi Sabeti Moghaddam, Zeinab Dehghani, Maneeha Rani, Koorosh Aslansefat, Bhupesh Kumar Mishra, Rameez Raja Kureshi, and Dhavalkumar Thakker.** “Explainable Knowledge Graph Retrieval-Augmented Generation (KG-RAG) with KG-SMILE.” arXiv:2509.03626, 2025.

KG-SMILE examines the contribution of retrieved entities, relations, graph paths, and contextual evidence to the answer generated by a knowledge-graph retrieval-augmented generation system. It is intended to expose irrelevant retrieval, unsupported reasoning paths, and excessive dependence on individual graph elements.

[Abstract](https://arxiv.org/abs/2509.03626) · [PDF](https://arxiv.org/pdf/2509.03626)

??? note "BibTeX"

    ```bibtex
    @misc{moghaddam2025kgsmile,
      author        = {Zahra Zehtabi Sabeti Moghaddam and Zeinab Dehghani
                       and Maneeha Rani and Koorosh Aslansefat
                       and Bhupesh Kumar Mishra and Rameez Raja Kureshi
                       and Dhavalkumar Thakker},
      title         = {Explainable Knowledge Graph Retrieval-Augmented
                       Generation ({KG-RAG}) with {KG-SMILE}},
      year          = {2025},
      eprint        = {2509.03626},
      archivePrefix = {arXiv},
      primaryClass  = {cs.AI},
      doi           = {10.48550/arXiv.2509.03626},
      url           = {https://arxiv.org/abs/2509.03626}
    }
    ```

## Concept-based explainability

### ConceptSMILE

**Mohadeseh Mollapour, Koorosh Aslansefat, Zeinab Dehghani, Bhupesh Kumar Mishra, Tejal Shah, and Zhibao Mian.** “ConceptSMILE: Auditing the Trustworthiness of Concept-Based Explainable AI.” arXiv:2607.09649, 2026.

ConceptSMILE audits whether higher-level, human-understandable concepts provide explanations that are faithful to the model, locally representative, stable under small changes, and consistent across related inputs. This is particularly relevant where explanations must be meaningful to domain experts as well as technically aligned with model behaviour.

[Abstract](https://arxiv.org/abs/2607.09649) · [PDF](https://arxiv.org/pdf/2607.09649)

??? note "BibTeX"

    ```bibtex
    @misc{mollapour2026conceptsmile,
      author        = {Mohadeseh Mollapour and Koorosh Aslansefat
                       and Zeinab Dehghani and Bhupesh Kumar Mishra
                       and Tejal Shah and Zhibao Mian},
      title         = {{ConceptSMILE}: Auditing the Trustworthiness of
                       Concept-Based Explainable {AI}},
      year          = {2026},
      eprint        = {2607.09649},
      archivePrefix = {arXiv},
      primaryClass  = {cs.AI},
      doi           = {10.48550/arXiv.2607.09649},
      url           = {https://arxiv.org/abs/2607.09649}
    }
    ```

## Related thesis

### Addressing Explainability of Generative AI Using SMILE

**Zeinab Dehghani.** “Addressing Explainability of Generative AI Using SMILE: Statistical Model-Agnostic Interpretability with Local Explanations.” MSc thesis, University of Hull, September 2025. Also available as arXiv:2602.01206.

The thesis consolidates the use of SMILE for generative artificial intelligence, particularly large language models and instruction-based image-editing systems. It is listed separately because it is a thesis rather than a peer-reviewed journal or conference publication.

[Abstract](https://arxiv.org/abs/2602.01206) · [PDF](https://arxiv.org/pdf/2602.01206)

??? note "BibTeX"

    ```bibtex
    @mastersthesis{dehghani2025generativeaismile,
      author = {Zeinab Dehghani},
      title  = {Addressing Explainability of Generative {AI} Using
                {SMILE}: Statistical Model-Agnostic Interpretability
                with Local Explanations},
      school = {University of Hull},
      type   = {{MSc} thesis},
      year   = {2025},
      month  = {September},
      note   = {Also available as arXiv:2602.01206},
      doi    = {10.48550/arXiv.2602.01206},
      url    = {https://arxiv.org/abs/2602.01206}
    }
    ```

!!! note "Maintaining citation metadata"
    Use the DOI for formally published work and the arXiv identifier for current preprints. When a preprint receives a peer-reviewed publication record, retain the arXiv link for version history but update the preferred citation to the final publisher metadata.
