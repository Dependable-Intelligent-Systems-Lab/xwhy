# Install XWhy

XWhy requires **Python 3.12 or later**.

## Core installation

```
pip install xwhy
```

Using `uv`:

```
uv add xwhy
```

## Optional cloud dependencies

Install only the provider integrations required by your project:

```
pip install "xwhy[vertex]"
pip install "xwhy[aws,bedrock]"
pip install "xwhy[all]"
```

Using `uv`:

```
uv add "xwhy[vertex]"
uv add "xwhy[aws,bedrock]"
uv add "xwhy[all]"
```

Protect credentials

Store API keys in environment variables or a local `.env` file. Never commit credentials to a repository, notebook, documentation page, or issue.

## Verify the installation

```
import xwhy

print(xwhy.__all__)
```

Next, [choose an explainer](https://dependable-intelligent-systems-lab.github.io/xwhy/getting-started/choosing-an-explainer/index.md).
