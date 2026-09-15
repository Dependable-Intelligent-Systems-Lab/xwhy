# Image Classification Explainer

Image classification models can be difficult to interpret because their predictions depend on complex visual patterns distributed across an image.

The **XWhy Image Classification Explainer** provides a model-agnostic way to examine an image classifier as a black-box system. It perturbs the input image, observes how the model prediction changes, and estimates which regions or segments of the image had the strongest local influence on the generated output.

This guide starts with the simplest setup using a built-in classification model. Advanced configuration for custom PyTorch models, embedding engines, segmentation models, and configuration objects is provided later.

> **Important:** Only PyTorch implementations are supported. You may pass only PyTorch models for explainability.

______________________________________________________________________

## Quick Start: Explain with a Built-in Model

For a first test, you only need a short Python script. The explainer can load a pre-defined classification model, apply its standard preprocessing, and generate a local explanation.

### 1. Run a Basic Explanation

The following example uses the default configuration and explains a local image:

```
from xwhy import ImageClassificationExplainer
import xwhy

try:
    explainer = ImageClassificationExplainer(
        use_model_preprocess=True,
        use_embedding_model=True,
        use_segmentation_model=True,
    )
    # or use `explainer.run`
    result = explainer.explain(instance="cat-and-dog.jpg")
    print(result.metrics)
    print("Explanation successful!")

    # Superpixel-level heatmap
    xwhy.plots.image_heatmap(result)

    # Additional image visualisation
    xwhy.plots.image(result)

except Exception as e:
    print(f"Error during pipeline execution: {e}")
```

### 2. Using an `ImageClassificationConfig` Object

You can also centralize settings in an `ImageClassificationConfig` instance and pass it to the `ImageClassificationExplainer` constructor as a single `config` parameter. This avoids supplying individual keyword arguments for every option:

```
from xwhy.core import ImageClassificationConfig
from xwhy import ImageClassificationExplainer
import xwhy

image_classification_config = ImageClassificationConfig(
    use_model_preprocess=True,
    use_embedding_model=False,
    embedding_type="dinov2",
    classification_type="inception_v3",

    # Custom Model
    custom_model=None,
    custom_preprocess=None,
    categories=None,

    use_segmentation_model=True,
    segmentation_type="deeplabv3_resnet101",
    device="cpu",
    seed=222,
    kernel_size=4,
    max_dist=200,
    ratio=0.2,
    num_perturb=150,
    distance_type="wasserstein",
    surrogate_type="lime",
    use_best_surrogate=True,
    num_top_features=4,
    num_top_predictions=5,
)

try:
    explainer = ImageClassificationExplainer(config=image_classification_config)
    # or use `explainer.run`
    result = explainer.explain(instance="cat-and-dog.jpg")
    print(result.metrics)
    print("Explanation successful!")

    # Superpixel-level heatmap
    xwhy.plots.image_heatmap(result)

    # Additional image visualisation
    xwhy.plots.image(result)

except Exception as e:
    print(f"Error during pipeline execution: {e}")
```

### 3. Read the Result

The explanation highlights image regions according to their estimated influence on the model prediction.

The returned `result` object also contains evaluation metrics that can be used to examine the quality and reliability of the local explanation. These metrics should be interpreted as evidence about the explanation produced by XWhy, rather than as direct access to the classifier’s internal decision process.

______________________________________________________________________

## Additional Explanation Plots

After generating a valid `result`, you can use the following visualisations:

```
# Superpixel-level heatmap
xwhy.plots.image_heatmap(result)

# Additional image visualisation
xwhy.plots.image(result)
```

______________________________________________________________________

## Using Pre-defined Classification Models

XWhy provides several ready-to-use classification models. Pass the desired identifier via the `classification_type` argument (or the corresponding field in `ImageClassificationConfig`):

- `inception_v3`
- `resnet18`
- `resnet50`
- `mobilenet_v3`
- `vit_base`

When using a built-in model you may also enable the model’s standard preprocessing pipeline with `use_model_preprocess=True`.

______________________________________________________________________

## Using a Custom PyTorch Model

You may supply your own PyTorch classification model together with a matching preprocessing pipeline. Two typical patterns are shown below.

### Simple Custom Model

```
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

from xwhy.explainers.image import ImageClassificationExplainer
import xwhy


# 1. Define the custom model
class CustomVisionModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        # Assuming a 32x32 input image, the size becomes 8x8 after two pooling steps
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the tensor for the linear layers
        x = torch.flatten(x, 1) 

        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        return logits

# 2. Create an instance of the custom model
my_custom_model = CustomVisionModel(num_classes=10)

# 3. Define a custom preprocessing pipeline for this model
# This model expects 32x32 inputs, so resizing is mandatory
my_preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    # CIFAR-10 dataset normalization values (as an example)
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]) 
])

# 4. Initialize the explainer with the new custom model
# For faster testing, you can disable the segmentation and embedding models
explainer = ImageClassificationExplainer(
    custom_model=my_custom_model,
    custom_preprocess=my_preprocess,
    use_embedding_model=False, 
    use_segmentation_model=False,
    device="cpu" 
)

image_path = "./cat-and-dog.jpg" 
result = explainer.explain(instance=image_path)
print(result.metrics)

print("Explanation Finished!")
print("Top Feature Indices:", result.top_features)

# Superpixel-level heatmap
xwhy.plots.image_heatmap(result)

# Additional image visualisation
xwhy.plots.image(result)
```

### Well-known Pre-trained Model (ResNet-18)

```
import urllib.request
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from xwhy.explainers.image import ImageClassificationExplainer
import xwhy

# 1. Load a well-known pre-trained PyTorch model (e.g., ResNet18)
# We use eval() mode since we are performing inference, not training.
weights = ResNet18_Weights.IMAGENET1K_V1
my_custom_model = resnet18(weights=weights)
my_custom_model.eval()

# 2. Define the exact preprocessing pipeline required by this model
# ImageNet models typically require resizing to 256, center cropping to 224, 
# and specific mean/std normalization values.
my_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Fetch ImageNet category names for human-readable logs
# This validates the 'categories' argument and MockWeights we added earlier.
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
try:
    response = urllib.request.urlopen(url)
    imagenet_categories = [line.decode("utf-8").strip() for line in response.readlines()]
except Exception:
    # Fallback to dynamic categories (Class 0, Class 1, ...) if offline
    imagenet_categories = None

# 4. Initialize the Explainer with our custom ResNet18 model
# By passing custom_model, we bypass the default factory and test our wrapper.
explainer = ImageClassificationExplainer(
    custom_model=my_custom_model,
    custom_preprocess=my_preprocess,
    categories=imagenet_categories,  # Pass the fetched class names here or use `weights.meta["categories"]`
    use_embedding_model=False,       # Disabled for faster CPU testing
    use_segmentation_model=False,    # Disabled to test the evaluation skip logic
    device="cpu" 
)

# 5. Run the explanation on a test image
image_path = "./cat-and-dog.jpg" 
result = explainer.explain(
    instance=image_path,
    ground_truth_mask=None  # Passing None to skip the ground-truth evaluation safely
)

print("\n--- Explanation Finished ---")
print("Metrics:", result.metrics)
print("Top Feature Indices:", result.top_features)

# Superpixel-level heatmap
xwhy.plots.image_heatmap(result)

# Additional image visualisation
xwhy.plots.image(result)
```

______________________________________________________________________

## Embedding and Segmentation Models

### Embedding Engine

XWhy currently supports a single local embedding engine for measuring changes between perturbed image representations:

- `dinov2`

Enable it with `use_embedding_model=True` and set `embedding_type="dinov2"` (or leave the default).

### Segmentation Models

When a ground-truth mask is not available, XWhy can generate super-pixel or semantic segments with one of the following models. Pass the identifier via `segmentation_type`:

- `deeplabv3_resnet101`
- `deeplabv3_resnet50`
- `deeplabv3_mobilenet_v3_large`
- `fcn_resnet50`
- `lraspp_mobilenet_v3_large`

If you already possess a mask for the image, supply it directly to the `explain` method through the `ground_truth_mask` argument. In that case the internal segmentation model is not required.

______________________________________________________________________

## Distance Metrics

Seven distance metrics are available for comparing original and perturbed representations. Select one via the `distance_type` argument (or the corresponding field in `ImageClassificationConfig`):

- `cosine`
- `wasserstein`
- `ks`
- `cramer_von_mises`
- `anderson_darling`
- `kuiper`
- `dts`

______________________________________________________________________

## Complete End-to-End Example

The following example creates the explainer with a built-in model, generates an explanation, inspects the returned metrics, and displays the available plots:

```
from xwhy import ImageClassificationExplainer
import xwhy

try:
    explainer = ImageClassificationExplainer(
        use_model_preprocess=True,
        use_embedding_model=True,
        use_segmentation_model=True,
        classification_type="resnet50",
        device="cpu",
    )

    result = explainer.explain(instance="cat-and-dog.jpg")

    print(result.metrics)
    print("Explanation generated successfully.")
    print("Top Feature Indices:", result.top_features)

    # Superpixel-level heatmap
    xwhy.plots.image_heatmap(result)

    # Additional image visualisation
    xwhy.plots.image(result)

except Exception as error:
    print(f"Error during the explanation pipeline: {error}")
```

______________________________________________________________________

## Common Setup Problems

### Model Not Found or Incompatible

Confirm that:

- the selected `classification_type` is one of the supported identifiers, or
- a valid PyTorch model is supplied via `custom_model` together with a matching `custom_preprocess` pipeline.

Only PyTorch models are accepted.

### Segmentation or Embedding Error

Confirm that the chosen `segmentation_type` or `embedding_type` is supported and that the required model weights can be downloaded or loaded on the selected `device`.

### Mask Handling

If you intend to use a custom mask, pass it through the `ground_truth_mask` parameter of the `explain` method. Leaving the parameter as `None` causes the explainer to rely on the configured segmentation model (if enabled).

______________________________________________________________________

## Interpretation Note

XWhy produces a local, perturbation-based approximation of region influence. It can help identify which parts of an image are associated with changes in a particular classification decision. It does not expose the model’s exact internal reasoning process or prove that a highlighted region was the sole cause of the predicted class.
