---
title: Install XWhy
description: Install XWhy with pip or uv and add optional cloud-provider dependencies for LLM explainability.
---

# Install XWhy

XWhy requires **Python 3.12 or later**.

## Core installation

```bash
pip install xwhy
```

Using `uv`:

```bash
uv add xwhy
```

## Optional cloud dependencies

Install only the provider integrations required by your project:

```bash
pip install "xwhy[vertex]"
pip install "xwhy[aws,bedrock]"
pip install "xwhy[all]"
```

Using `uv`:

```bash
uv add "xwhy[vertex]"
uv add "xwhy[aws,bedrock]"
uv add "xwhy[all]"
```

!!! warning "Protect credentials"
    Store API keys in environment variables or a local `.env` file. Never commit credentials to a repository, notebook, documentation page, or issue.

## Verify the installation

```python
import xwhy

print(xwhy.__all__)
```

Next, [choose an explainer](choosing-an-explainer.md).
