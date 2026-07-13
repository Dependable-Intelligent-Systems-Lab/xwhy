# Large Language Model (LLM) Explainer

Large Language Models (LLMs) can be difficult to interpret because their outputs depend on many interacting words, tokens, and contextual relationships.

The **XWhy LLM Explainer** provides a model-agnostic way to examine an LLM as a black-box system. It perturbs the input text, observes how the model response changes, and estimates which input words or tokens had the strongest local influence on the generated output.

This guide starts with the simplest setup for explaining an OpenAI model. Advanced configuration for other cloud providers, routers, and local models is provided later.

---

## Quick Start: Explain an OpenAI Model

For a first test, you only need:

1. An LLM API provider key
2. A credential setup approach (either a `.env` file or direct notebook/runtime configuration)
3. A short Python script

### 1. Create a Simple `.env` File

Create a file named `.env` in the root directory of your project and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here

```

> Keep API keys private. Do not commit the `.env` file to GitHub or include it in shared code.

### 2. Alternative: Notebook & Runtime Configuration (No `.env` Required)

If you are working in an interactive environment like Google Colab or Jupyter Notebooks, managing a physical `.env` file can be cumbersome. XWhy provides two flexible alternatives to configure your credentials on the fly directly inside your code.

#### Approach A: Modifying Global Settings

You can dynamically assign values to the global `settings` object immediately after importing `xwhy`. This updates the internal config registry before the explainer is initialized:

```python
import xwhy

# Manually set keys at runtime (Overrides .env)
xwhy.settings.openai_api_key = "sk-proj-your_key_here"
xwhy.settings.gemini_api_key = "your_gemini_key_here"

```

#### Approach B: Inline Arguments via LLMExplainer (`kwargs`)

You can pass credentials and client settings directly into the `LLMExplainer` constructor. These parameters bypass the global state and are forwarded straight to the provider's underlying engine:

```python
from xwhy import LLMExplainer

# Pass parameters directly into the explainer constructor
explainer = LLMExplainer(
    provider="openai",
    api_key="sk-proj-your_key_here",
    use_best_surrogate=True
)

```

> ⚠️ **Critical Requirement on Parameter Names:** When passing credentials directly into `LLMExplainer`, the key arguments **must match the native parameter names expected by the underlying provider's official SDK client**, not XWhy's global setting aliases.
> For example, use **`api_key`** for OpenAI, Anthropic, or Gemini, but you must use **`token`** when instantiating a Hugging Face client, as shown below:

```python
# For OpenAI/Anthropic/Gemini, the SDK client expects 'api_key'
openai_explainer = LLMExplainer(provider="openai", api_key="sk-...")

# For Hugging Face, the native InferenceClient SDK expects 'token'
hf_explainer = LLMExplainer(provider="huggingface", token="hf_...")

```

### 3. Run a Basic Explanation

The following example explains the input text using an OpenAI model and displays a token-level heatmap:

```python
import xwhy
from xwhy import LLMExplainer

# Required for interactive plots in notebook environments
xwhy.plots.initjs()

try:
    # Create an explainer for an OpenAI model
    # (Credentials will look up your .env, xwhy.settings, or explicit kwargs)
    explainer = LLMExplainer(
        provider="openai",
        use_best_surrogate=True,
    )

    # Explain the model response for the supplied input
    result = explainer.explain(
        instance="Machine learning is fascinating.",
        model_name="gpt-5-nano",
        fidelity_plot=False,
    )

    # Print the explanation-quality metrics
    print(result.metrics)

    # Display the fidelity plot
    result.plot()

    # Display the token-level explanation
    xwhy.plots.text_heatmap(result)

except Exception as error:
    print(f"The explanation could not be generated: {error}")

```

### 4. Read the Result

The explanation highlights words or tokens according to their estimated influence on the model response.

The returned `result` object also contains evaluation metrics that can be used to examine the quality and reliability of the local explanation. These metrics should be interpreted as evidence about the explanation produced by XWhy, rather than as direct access to the LLM's internal reasoning process.

---

## Additional Explanation Plots

After generating a valid `result`, you can use the following visualisations:

```python
xwhy.plots.bar(result)
xwhy.plots.waterfall(result)
xwhy.plots.text(result)
xwhy.plots.force(result)
xwhy.plots.decision(result)

```

A complete example that generates all supported plots is provided later in this guide.

---

## Advanced Configuration

Use this section when you want to work with providers other than OpenAI, use proxy services, connect to cloud platforms, or run local models.

### 1. Full Environment Variable Template

Create a `.env` file in the root directory of your project. You only need to populate the variables required by the providers you intend to use.

```env
############################
# XWhy Provider Configuration
############################

# OpenAI
OPENAI_API_KEY=

# Google Gemini
GEMINI_API_KEY=

# Anthropic Claude
ANTHROPIC_API_KEY=

# Z.AI
ZAI_API_KEY=

# Groq
GROQ_API_KEY=

# Cohere
COHERE_API_KEY=

# Fireworks AI
FIREWORKS_API_KEY=

# Grok (xAI)
GROK_API_KEY=

# OpenRouter
OPENROUTER_API_KEY=

# ByteDance
BYTEDANCE_API_KEY=

# LM Studio
LMSTUDIO_API_KEY=
LMSTUDIO_BASE_URL=

# Azure OpenAI
AZURE_API_KEY=
AZURE_API_VERSION="2024-02-01"
AZURE_ENDPOINT=

# AWS / Bedrock
AWS_ACCESS_KEY=
AWS_SECRET_KEY=
AWS_SESSION_TOKEN=
AWS_REGION="us-east-1"
ANTHROPIC_AWS_WORKSPACE_ID=

# Google Cloud: Gemini and Anthropic Vertex
GCP_PROJECT=
GCP_LOCATION="us-central1"

# Microsoft Foundry: Anthropic
ANTHROPIC_FOUNDRY_API_KEY=
ANTHROPIC_FOUNDRY_RESOURCE=

# Hugging Face
HUGGINGFACE_TOKEN=

############################
# Embedding Cache
############################

EMBEDDING_CACHE_DIR=~/.cache/xwhy/embeddings

```

### 2. Select a Provider in Python

Change the `provider` value when creating the explainer:

```python
from xwhy import LLMExplainer

explainer = LLMExplainer(
    provider="openai",
    use_best_surrogate=True,
)

```

For another supported provider, replace `"openai"` with the provider identifier required by XWhy and ensure that the corresponding environment variables are configured.

### 3. Embedding Storage Configuration

The `EMBEDDING_CACHE_DIR` variable specifies where local text-embedding files are stored.

* **Pre-cached models:** If the required embedding files already exist in this directory, XWhy can load them locally.
* **Automatic fallback:** If the required files are missing, XWhy can download and cache them for future explanation runs.

The configured directory must be writable by the current user.

---

## Supported Ecosystems

### Supported LLM Providers

XWhy supports commercial APIs, cloud platforms, proxy routers, and local inference services, including:

* OpenAI
* Google Gemini
* Anthropic
* Hugging Face
* Z.AI
* Groq
* Cohere
* Fireworks AI
* Grok (xAI)
* OpenRouter
* Ollama
* LM Studio
* ByteDance
* Azure OpenAI
* GCP Gemini
* Anthropic Bedrock
* Anthropic Bedrock Mantle
* Anthropic AWS
* Anthropic Vertex
* Anthropic Foundry

### Supported Embedding Engines

XWhy currently supports the following local embedding engines for measuring changes between perturbed text representations:

* Word2Vec
* GloVe
* Paragram-sl
* Paragram-WS

---

## Complete End-to-End Example

The following example creates the explainer, generates an explanation, prints the metrics, and displays all available diagnostic plots.

```python
import xwhy
from xwhy import LLMExplainer

# Initialize interactive JavaScript-based plots
xwhy.plots.initjs()

try:
    # Configure the target provider and surrogate-selection behaviour
    explainer = LLMExplainer(
        provider="openai",
        use_best_surrogate=True,
    )

    # Generate a local explanation for the selected model and input
    result = explainer.explain(
        instance="Machine learning is fascinating.",
        model_name="gpt-5-nano",
    )

    # Inspect the explanation metrics
    print(result.metrics)
    print("Explanation generated successfully.")

    # Token-level heatmap
    xwhy.plots.text_heatmap(result)

    # Additional diagnostic plots
    xwhy.plots.bar(result)
    xwhy.plots.waterfall(result)
    xwhy.plots.text(result)
    xwhy.plots.force(result)
    xwhy.plots.decision(result)

except Exception as error:
    print(f"Error during the explanation pipeline: {error}")

```

---

## Common Setup Problems

### API Key Not Found

Confirm that:

* the file is named exactly `.env`;
* it is located in the project root directory; and
* the relevant API key variable is populated.

### Model Access Error

The selected model must be available through your provider account. If the model is unavailable, replace `model_name` with a model that your account can access.

### Embedding Download or Cache Error

Confirm that the path assigned to `EMBEDDING_CACHE_DIR` exists or can be created, and that the current user has permission to write to it.

---

## Interpretation Note

XWhy produces a local, perturbation-based approximation of input influence. It can help identify which words or tokens are associated with changes in a particular model response. It does not expose private chain-of-thought reasoning, recover the model's exact internal decision process, or prove that a highlighted token was the sole cause of the generated output.
