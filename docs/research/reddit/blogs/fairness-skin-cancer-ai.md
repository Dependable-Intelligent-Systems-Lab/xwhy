---
title: Can We Trust AI for Skin Cancer?
description: Why fairness, reliability and human oversight matter when AI is used with skin images.
---

[← All research and insights](../blogs.md){ .xwhy-blog-back }

# Can we trust AI for skin cancer?

![An original illustration of a medical image review screen](../../../assets/images/blogs/skin-cancer.svg)

*Responsible AI · Research note based on [the Hull Pint of Science event report](https://www.responsibleaihull.com/post/pint-of-science-trustworthy-ai-can-we-trust-ai-for-skin-cancer-are-they-fair) by Dhaval Thakker.*

At a Hull Pint of Science event in May 2024, Kuniko Paxton and Koorosh Aslansefat discussed fairness and reliability in AI systems for skin cancer detection. Their public discussion raises a question that applies well beyond this one task: who benefits from a system's reported accuracy, and who might face a greater risk of error?

## Accuracy needs context

An overall test score can conceal differences across skin tones, image capture conditions and patient groups. A careful evaluation should report performance across relevant groups and examine the data behind those results. Group definitions, sample sizes and clinical context matter: a small or unrepresentative test set cannot establish that a system works equally well for everyone.

Explanations can help investigators ask whether a model is responding to plausible image regions or irrelevant cues. They cannot, by themselves, establish clinical validity or fairness. A visually attractive heatmap might still be unstable or fail to reflect what drives the prediction.

## What responsible use requires

Before clinical use, an AI system needs evaluation with data that reflects the intended patients and setting, an account of uncertainty and failure modes, and clear human responsibility for decisions. Testing should include cases where image quality is poor or a case differs from the development data. If an explanation changes sharply after a harmless change to the image, that is a reason to investigate, not a basis for greater confidence.

XWhy's [guidance on the limits of local explanations](../../../concepts/limitations.md) describes some of the checks needed when explanations are used in high-impact settings.

**Original event report:** [Pint of Science! Trustworthy AI: Can we trust AI for skin cancer? Are they fair?](https://www.responsibleaihull.com/post/pint-of-science-trustworthy-ai-can-we-trust-ai-for-skin-cancer-are-they-fair). This page is an original XWhy summary of the themes, not a clinical recommendation.
