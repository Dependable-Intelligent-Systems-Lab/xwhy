---
title: Configure LLM Providers in XWhy
description: Configure credentials and provider selection for the XWhy LLM explainer without exposing API keys.
---

# Configure LLM providers

The LLM explainer supports configuration through environment variables, runtime settings, constructor arguments, or an `LLMConfig` object.

[Read the complete provider setup guide](../llm_explainer.md#advanced-configuration)

## Security requirements

- Never commit `.env` files containing credentials.
- Use repository secrets for automated tests and documentation builds.
- Do not print keys in notebook outputs or exception messages.
- Use provider accounts and models that you are authorised to access.
- Review cost and rate-limit implications before increasing perturbation counts.
