import os
import logging
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import AutoImageProcessor, AutoModel

from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import wasserstein_distance
from gensim.models import KeyedVectors
import gensim.downloader as api
from img2img_turbo import run_inference_paired


# --------------------------------------------------------------------------- #
# Configure logger
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Word2Vec model
# --------------------------------------------------------------------------- #

# TODO: Improving and using better Word2Vec Models and also giving user an option
# to have preference on this


def load_google_news_vectors():
    # First try gensim downloader (fastest & simplest)
    try:
        logger.info("Trying to load 'word2vec-google-news-300' via gensim.api...")
        return api.load("word2vec-google-news-300")
    except Exception:
        logger.error("Gensim API unavailable — downloading manually...")

    # Manual fallback (public mirror)
    url = "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
    local_path = "GoogleNews-vectors-negative300.bin.gz"
    extracted_path = "GoogleNews-vectors-negative300.bin"

    # Download if missing
    if not os.path.exists(local_path):
        import urllib.request
        logger.info("Downloading GoogleNews vectors...")
        urllib.request.urlretrieve(url, local_path)

    # Extract if needed
    if not os.path.exists(extracted_path):
        import gzip
        import shutil
        logger.info("Extracting .gz file...")
        with gzip.open(local_path, "rb") as f_in:
            with open(extracted_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    logger.info("Loading Word2Vec model...")
    return KeyedVectors.load_word2vec_format(extracted_path, binary=True)


# --------------------------------------------------------------------------- #
# Perturbations
# --------------------------------------------------------------------------- #

def apply_binary_mask(
    text_list: list[str],
    mask: tuple[int, ...],
    rng: np.random.Generator
) -> list[str]:
    """Apply a binary mask to a list of words and ensure minimum word count.

    The mask selects words where a `1` indicates inclusion. If the mask
    includes fewer than two valid words, additional random words from the
    original text are added to preserve semantic stability.

    Args:
        text_list (list[str]): Tokenized original text.
        mask (tuple[int, ...]): Binary mask indicating selected words.
        rng (np.random.Generator): Random generator for selecting fallback
            words.

    Returns:
        list[str]: The perturbed list of words.
    """
    selected = [
        w for w, flag in zip(text_list, mask)
        if flag == 1 and w.strip() != ""
    ]

    if len(selected) < 2:
        needed = 2 - len(selected)
        candidates = [w for w in text_list if w.strip() != ""]
        extra = rng.choice(candidates, needed, replace=False)
        result = list(set(selected + list(extra)))
    else:
        result = list(set(selected))

    return result


def generate_perturbations(
    text: str,
    num_perturb: int = 64,
    seed: int = 1024
) -> tuple[list[str], list[tuple]]:
    """Generate unique binary perturbations and perturbed text versions.

    Args:
        text (str): Original input text.
        num_perturb (int): Number of perturbations to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple[list[str], list[tuple]]:
            - List of perturbed texts (Responses)
            - List of binary perturbation vectors (Perturbations)
    """
    words = text.split()
    text_list = words.copy()
    num_words = len(words)

    rng = np.random.default_rng(seed)

    responses = []
    perturbations = []
    unique_perturbations = set()

    attempts = 0
    max_attempts = num_perturb * 10

    # Loop for unique perturbations only
    while len(unique_perturbations) < num_perturb and attempts < max_attempts:
        p = tuple(rng.binomial(1, 0.5, size=num_words))

        if p not in unique_perturbations and sum(p) > 0:
            unique_perturbations.add(p)
            perturbed_txt = apply_binary_mask(text_list, p, rng)
            corpus = " ".join(perturbed_txt)

            responses.append(corpus)
            perturbations.append(p)

            logger.info(
                "Perturbation: %s, Perturbed Text: %s", p, corpus
            )

        attempts += 1

    # If more perturbations are needed, allow repeats by sampling
    # from unique_perturbations
    while len(responses) < num_perturb:
        p = rng.choice(list(unique_perturbations))
        perturbed_txt = apply_binary_mask(text_list, p, rng)
        corpus = " ".join(perturbed_txt)

        responses.append(corpus)
        perturbations.append(p)

        logger.info(
            "Perturbation (reused): %s, Perturbed Text: %s", p, corpus
        )

    return responses, perturbations


# --------------------------------------------------------------------------- #
# Generate image and embedding
# --------------------------------------------------------------------------- #

def extract_image_embedding(
    image: Image.Image,
    processor: Any,
    model: Any
) -> np.ndarray:
    """Extract an image embedding using a DINOv2 model.

    Args:
        image (PIL.Image.Image): Input image.
        processor (Any): DINOv2 image processor.
        model (Any): DINOv2 model.

    Returns:
        np.ndarray: Extracted embedding vector.
    """
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    emb = outputs.last_hidden_state.mean(dim=1)
    emb = emb.squeeze().cpu().numpy()
    return emb


def generate_image(
    input_image_path: str,
    prompt: str,
    output_dir: str,
    model_name: str,
    low_threshold: Optional[int] = None,
    high_threshold: Optional[int] = None,
    seed: Optional[int] = None,
) -> str:
    """Generate an image using the inference_paired pipeline.

    Args:
        input_image_path (str): Path to the original image.
        prompt (str): Text instruction for image editing.
        output_dir (str): Directory to store generated images.
        model_name (str): Name of the Pix2Pix-style model.
        low_threshold (int | None): Canny low threshold.
        high_threshold (int | None): Canny high threshold.
        seed (int | None): Random seed.

    Returns:
        str: Path to the most recently generated image.

    Raises:
        FileNotFoundError: If no output files are found.
        Exception: For unexpected internal errors.
    """
    try:
        run_inference_paired(
            model_name=model_name,
            input_image=input_image_path,
            prompt=prompt,
            output_dir=output_dir,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            seed=seed,
        )

        files = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
        ]

        if not files:
            raise FileNotFoundError(
                f"No files found in directory: {output_dir}"
            )

        return max(files, key=os.path.getctime)

    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error during image generation: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# Pix2Pix Pipeline
# --------------------------------------------------------------------------- #

def generate_image_and_embedding(
    input_image_path: str,
    prompt: str,
    output_dir: str,
    model_name: str,
    low_threshold: int,
    high_threshold: int,
    seed: int,
    processor: Any,
    model: Any
) -> Tuple[str, Image.Image, np.ndarray]:
    """Generate an edited image and extract its embedding.

    Args:
        input_image_path (str): Path to original image.
        prompt (str): Text prompt used for image generation.
        output_dir (str): Directory to save generated image.
        model_name (str): Name of the Pix2Pix model.
        low_threshold (int): Canny low threshold.
        high_threshold (int): Canny high threshold.
        seed (int): Random seed for reproducibility.
        processor (Any): DINOv2 processor.
        model (Any): DINOv2 model.

    Returns:
        Tuple[str, Image.Image, np.ndarray]:
            - Path to generated image.
            - Generated PIL image.
            - Extracted DINOv2 embedding as a NumPy array.
    """
    gen_path = generate_image(
        input_image_path=input_image_path,
        prompt=prompt,
        output_dir=output_dir,
        model_name=model_name,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        seed=seed,
    )

    gen_img = Image.open(gen_path)
    gen_emb = extract_image_embedding(gen_img, processor, model)
    # TODO: Maybe need to remove the `np.asarray`
    return gen_path, gen_img, np.asarray(gen_emb)


def process_single_perturbation(
    perturbation: Tuple[int, ...],
    text_list: List[str],
    rng: np.random.Generator,
    input_image_path: str,
    output_dir: str,
    model_name: str,
    low_threshold: int,
    high_threshold: int,
    seed: int,
    processor,
    model,
    orig_embedding: np.ndarray
) -> float:
    """Process a single perturbation and compute Wasserstein distance.

    Args:
        perturbation (tuple[int]): Binary perturbation mask.
        text_list (list[str]): Token list of original prompt.
        rng (np.random.Generator): Random generator.
        input_image_path (str): Path to original image.
        output_dir (str): Output directory for generated image.
        model_name (str): Pix2Pix model name.
        low_threshold (int): Canny low threshold.
        high_threshold (int): Canny high threshold.
        seed (int): Random seed.
        processor: DINOv2 processor.
        model: DINOv2 model.
        orig_embedding (np.ndarray): Embedding of original image.

    Returns:
        float: Wasserstein distance for this perturbation.
    """
    perturbed_txt = apply_binary_mask(text_list, perturbation, rng)
    corpus = " ".join(perturbed_txt)

    # Generate image and embedding
    _, gen_img, gen_emb = generate_image_and_embedding(
        input_image_path,
        corpus,
        output_dir,
        model_name,
        low_threshold,
        high_threshold,
        seed,
        processor,
        model
    )

    if gen_emb is None or orig_embedding is None:
        raise ValueError(
            "One or both embeddings are None. Check embedding extraction."
        )

    gen_emb = np.asarray(gen_emb)
    orig_emb = np.asarray(orig_embedding)

    if gen_emb.size == 0 or orig_emb.size == 0:
        raise ValueError(
            "Embeddings are empty. Cannot compute Wasserstein distance."
        )

    # Compute Wasserstein distance
    dist = wasserstein_distance(gen_emb, orig_emb)

    # Display result
    logger.info("Perturbation:")
    logger.info(f"Perturbed Text: {corpus}")
    logger.info(f"Wasserstein distance (generated vs orig): {dist}")

    plt.figure(figsize=(8, 8))
    plt.imshow(gen_img)
    plt.title(f"Perturbed Text: {corpus}", fontsize=12)
    plt.axis("off")
    plt.show()

    return dist


def process_all_perturbations(
    perturbations: List[Tuple[int, ...]],
    prompt: str,
    rng: np.random.Generator,
    input_image_path: str,
    output_dir: str,
    model_name: str,
    low_threshold: int,
    high_threshold: int,
    seed: int,
    processor: Any,
    model: Any,
    orig_embedding: np.ndarray,
    save_path: str = "WD_dists_generated_vs_orig.npy"
) -> List[Tuple[str, float]]:
    """Run Wasserstein computation for all perturbations.

    Args:
        perturbations (list[tuple[int]]): All generated masks.
        prompt (str): Original text prompt.
        rng (np.random.Generator): Random generator.
        input_image_path (str): Original image path.
        output_dir (str): Output directory.
        model_name (str): Pix2Pix model name.
        low_threshold (int): Canny low threshold.
        high_threshold (int): Canny high threshold.
        seed (int): Random seed.
        processor: DINOv2 processor.
        model: DINOv2 model.
        orig_embedding (np.ndarray): Original image embedding.
        save_path (str): File path to save Wasserstein distances.

    Returns:
        List[Tuple[str, float]]: List of (perturbation index as string, Wasserstein distance) pairs.
    """
    words = prompt.split()

    distances = []

    for p_idx, p in enumerate(perturbations):
        dist = process_single_perturbation(
            p,
            words,
            rng,
            input_image_path,
            output_dir,
            model_name,
            low_threshold,
            high_threshold,
            seed,
            processor,
            model,
            orig_embedding
        )
        distances.append((str(p_idx), dist))

    np.save(save_path, np.array(distances))
    logger.info("All generated embeddings and Wasserstein distances saved.")

    return distances


def run_pix2pix_pipeline(
    processor: Any,
    model: Any,
    prompt: str,
    input_image_path: str,
    output_dir: str,
    model_name: str,
    low_threshold: int,
    high_threshold: int,
    seed: int,
    perturbations: List[Tuple[int, ...]],
    orig_embedding: np.ndarray,
    save_path: str = "WD_dists_generated_vs_orig.npy"
) -> List[Tuple[str, float]]:
    """High-level pipeline to process perturbations and compute distances.

    Args:
        processor: DINOv2 processor.
        model: DINOv2 model.
        prompt (str): Input prompt.
        input_image_path (str): Path to original image.
        output_dir (str): Directory to save generated results.
        model_name (str): Pix2Pix model.
        low_threshold (int): Canny low threshold.
        high_threshold (int): Canny high threshold.
        seed (int): Random seed.
        perturbations (list[tuple[int]]): Perturbation masks.
        orig_embedding (np.ndarray): Original image embedding.
        save_path (str): Where to save computed distances.

    Returns:
        List[Tuple[str, float]]: List of (perturbation index as string, Wasserstein distance) pairs.
    """
    rng = np.random.default_rng(seed)

    return process_all_perturbations(
        perturbations,
        prompt,
        rng,
        input_image_path,
        output_dir,
        model_name,
        low_threshold,
        high_threshold,
        seed,
        processor,
        model,
        orig_embedding,
        save_path
    )


# --------------------------------------------------------------------------- #
# WMD-based similarity
# --------------------------------------------------------------------------- #

def compute_wmd_scores(model, original: str, responses: list) -> list:
    """Compute safe WMD distances between prompt and perturbed outputs.

    Args:
        model: Word2Vec model.
        original (str): prompt.
        responses (list): List of perturbed texts.

    Returns:
        list: List of (perturbed_text, distance).
    """
    scores = []
    for text in responses:
        dist = model.wmdistance(original, text)
        scores.append((text, dist))

    for text, dist in scores:
        logger.info("Perturbed Text: %s", text)
        logger.info("Distance Score: %.4f", dist)
        logger.info("-" * 50)

    return scores


def normalize_similarities(
    distances: list[tuple[str, float]],
    mode: str = "linear"
) -> list[tuple[str, float]]:
    """
    Convert distance values into similarity scores in range [0, 1].

    Supports two normalization strategies:

    1. 'linear':
        similarity = 1 - MinMax(distance)
        Recommended for regression-based explainability:
        Preserves linear proportionality and avoids non-linear distortion.

    2. 'inverse':
        similarity = MinMax(1 / (distance + ε))
        Matches original pix2pix notebook.
        Emphasizes small distances more aggressively (non-linear boost).

    Args:
        distances (list[tuple[str, float]]):
            List of (text, distance) pairs.
        mode (str):
            Normalization mode. One of {"linear", "inverse"}.

    Returns:
        list[tuple[str, float]]:
            List of (text, similarity) pairs.

    Raises:
        ValueError: If invalid mode is provided.
    """
    if mode not in ("linear", "inverse"):
        raise ValueError("mode must be one of: 'linear', 'inverse'")

    # Extract only the numeric distance values
    dist_values = np.array([d for _, d in distances], dtype=float)

    if len(dist_values) == 0:
        logger.error("Distance list is empty. Cannot normalize similarities.")
        return []

    # ==========================
    #   MODE: INVERSE
    # ==========================
    if mode == "inverse":
        epsilon = 1e-8
        inv = 1.0 / (dist_values + epsilon)

        min_v = inv.min()
        max_v = inv.max()

        if max_v == min_v:
            # All distances identical => all similarities identical
            sim_vals = np.ones_like(inv)
        else:
            sim_vals = (inv - min_v) / (max_v - min_v)

        # Pack results
        results = [(text, float(sim)) for (text, _), sim in zip(distances, sim_vals)]

        # Logging
        for text, sim in results:
            logger.info("Inverse Mode => Text: %s | Similarity: %.4f", text, sim)
            logger.info("-" * 50)

        return results

    # ==========================
    #   MODE: LINEAR
    # ==========================
    min_v = dist_values.min()
    max_v = dist_values.max()

    if max_v == min_v:
        # No variation in distances => give uniform similarity
        sim_vals = np.ones_like(dist_values)
    else:
        norm = (dist_values - min_v) / (max_v - min_v)
        sim_vals = 1.0 - norm  # invert because small distance => high similarity

    # Pack results
    results = [(text, float(sim)) for (text, _), sim in zip(distances, sim_vals)]

    # Logging
    for text, sim in results:
        logger.info("Linear Mode => Text: %s | Similarity: %.4f", text, sim)
        logger.info("-" * 50)

    return results


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #

# TODO: This part can be extended to cover other types of LIME, like BayLIME

def fit_weighted_regression(
    perturbations: list,
    similarities: list,
    wmd_scores: list
) -> tuple:
    """Fit weighted linear regression using WMD distances.

    Args:
        perturbations (list): List of binary perturbation vectors
            (Perturbations)
        similarities (list): List of (text, similarity).
        wmd_scores (list): List of (text, distance).

    Returns:
        tuple: (linear_model, coefficients, weights)
    """
    # Convert Perturbations to a NumPy array
    perturb_vecs = np.vstack(perturbations)  # Stack all perturbation vectors vertically

    dvals = np.array([d for _, d in wmd_scores])
    kernel_w = 0.25
    weights = np.sqrt(np.exp(-(dvals ** 2) / (kernel_w ** 2)))

    logger.debug("Weights:\n{weights}", weights)
    logger.debug("Perturbations:\n{perturbations}", perturb_vecs)

    y = np.array([s for _, s in similarities])

    linear_model = LinearRegression()
    linear_model.fit(perturb_vecs, y, sample_weight=weights)

    coeffs = linear_model.coef_
    logger.info("Regression coefficients: %s", coeffs)

    return linear_model, coeffs, weights


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_regression_metrics(
    linear_model: LinearRegression,
    coeffs: np.ndarray,
    weights: np.ndarray,
    similarities: list,
    perturbations: list
) -> dict:
    """Compute weighted regression metrics.

    Args:
        linear_model: Trained LinearRegression model.
        coeffs (np.ndarray): Regression coefficients.
        weights (np.ndarray): Sample weights.
        similarities (list): (text, similarity).
        perturbations (list): List of binary perturbation vectors.

    Returns:
        dict: Metrics including weighted R2, L1/L2 losses.
    """
    y_true = np.array([s for _, s in similarities])
    y_pred = linear_model.predict(perturbations).ravel()

    # Base regression metrics
    mse = mean_squared_error(y_true, y_pred, sample_weight=weights)
    mae = mean_absolute_error(y_true, y_pred, sample_weight=weights)

    # Mean loss
    mean_loss = abs(np.mean(y_true) - np.mean(y_pred))

    # L1 and L2 losses
    diff = y_true - y_pred
    mean_l1 = np.mean(np.abs(diff))
    mean_l2 = np.mean(diff ** 2)

    # Weighted L1 & L2
    n = len(y_true)
    weighted_l1 = np.sum(weights * np.abs(diff)) / n
    weighted_l2 = np.sum(weights * (diff ** 2)) / n

    # Weighted R² & Adjusted
    p = len(coeffs)
    f_mean = np.average(y_true, weights=weights)
    ss_tot = np.sum(weights * (y_true - f_mean) ** 2)
    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    weighted_r2 = 1 - ss_res / ss_tot

    weighted_adj_r2 = 1 - (1 - weighted_r2) * (n - 1) / (n - p - 1)

    return {
        "Mean Squared Error (MSE)": mse,
        "Mean Absolute Error (MAE)": mae,
        "Mean Loss (Lm)": mean_loss,
        "Mean L1 Loss": mean_l1,
        "Mean L2 Loss": mean_l2,
        "Weighted L1 Loss": weighted_l1,
        "Weighted L2 Loss": weighted_l2,
        "Weighted R-squared (R²ω)": weighted_r2,
        "Weighted Adjusted R-squared (R^²ω)": weighted_adj_r2,
    }


def print_metrics(metrics):
    print('-' * 100)
    print("Fidelity:")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print('-' * 100)


# --------------------------------------------------------------------------- #
# Heatmap visualization
# --------------------------------------------------------------------------- #

def plot_text_heatmap(
    words: list,
    scores: np.ndarray,
    title: str = "",
    width: float = 10.0,
    height: float = 0.4,
    verbose: int = 0,
    max_word_per_line: int = 20,
    word_spacing: int = 20,
    score_fontsize: int = 10,
    save_path: str | None = None
) -> None:
    """Plot a heatmap-like visualization over text tokens.

    Each token is shown inside a colored box based on its score, with the
    numeric score displayed underneath it.

    Args:
        words (list): List of text tokens.
        scores (np.ndarray): Array of per-token scores.
        title (str): Title shown on the plot.
        width (float): Figure width in inches.
        height (float): Figure height in inches.
        verbose (int): If 0, hide axes (clean output).
        max_word_per_line (int): Max number of tokens per visual line.
        word_spacing (int): Horizontal spacing between tokens.
        score_fontsize (int): Font size for numeric score labels.
        save_path (str | None): Optional save path.

    Returns:
        None
    """
    fig = plt.figure(figsize=(width, height))
    ax = plt.gca()
    ax.set_title(title, loc="left")

    # Color map normalization
    cmap = plt.cm.ScalarMappable(cmap=plt.cm.bwr)
    cmap.set_clim(0, 1)

    denom = np.max(np.abs(scores))
    if denom == 0:
        denom = 1e-8   # avoid division by zero
    normalized = 0.5 * scores / denom + 0.5

    canvas = ax.figure.canvas
    transform = ax.transData

    y = -0.2  # starting y

    for i, (word, score, ns) in enumerate(zip(words, scores, normalized)):
        r, g, b, _ = cmap.to_rgba(ns, bytes=True)
        color = f"#{r:02x}{g:02x}{b:02x}"

        # draw token
        txt = ax.text(
            0.0, y, word,
            bbox={
                "facecolor": color,
                "pad": 5.0,
                "linewidth": 1,
                "boxstyle": "round,pad=0.5"
            },
            transform=transform,
            fontsize=14
        )
        txt.draw(canvas.get_renderer())
        ex = txt.get_window_extent()

        # draw numeric score under token
        score_txt = ax.text(
            0.01,
            y - 1.0,
            f"{score:.2f}",
            transform=transform,
            fontsize=score_fontsize,
            ha="center"
        )
        score_txt.draw(canvas.get_renderer())

        # new transform for next token
        if (i + 1) % max_word_per_line == 0:
            y -= 2.5
            transform = ax.transData
        else:
            transform = transforms.offset_copy(
                txt._transform,
                x=ex.width + word_spacing,
                units="dots"
            )

    if verbose == 0:
        ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


# --------------------------------------------------------------------------- #
# High-level pipeline
# --------------------------------------------------------------------------- #

def run_pipline(
    prompt: str,
    input_image_path: str,
    model_name: str,
    output_dir: str = "output",
    low_threshold: int = 70,
    high_threshold: int = 200,
    seed: int = 1024,
    plot_heatmap: bool = True,
    mode: str = "linear"
) -> Dict[str, Any]:
    """Run the full interpretability pipeline for pix2pix text perturbations.

    This function:
      1. Sets random seeds for reproducibility.
      2. Loads DINOv2 and Word2Vec (Google News) models.
      3. Generates the edited image + embedding for the original prompt.
      4. Generates text perturbations.
      5. Runs the pix2pix pipeline for all perturbations
         and computes Wasserstein distances.
      6. Computes WMD scores between prompt and perturbed texts.
      7. Normalizes similarity values.
      8. Fits weighted linear regression on perturbation masks.
      9. Computes regression metrics.
     10. Optionally plots a heatmap over text tokens.

    Args:
        prompt (str): Input prompt describing the desired edit.
        input_image_path (str): Path to the input image.
        model_name (str): Name of the pix2pix-style model.
        output_dir (str): Directory for saving generated images.
        low_threshold (int): Canny edge low threshold.
        high_threshold (int): Canny edge high threshold.
        seed (int): Global random seed.
        plot_heatmap (bool): Whether to plot the text heatmap.
        mode (str): Normalization mode for similarity scores, one of {"linear", "inverse"}.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - orig_image_path (str)
            - responses (list[str])
            - wmd_scores (list)
            - similarities (list)
            - coefficients (np.ndarray)
            - weights (np.ndarray)
            - metrics (dict)
    """
    # logger.info("Setting seeds for reproducibility...")
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # logger.info("Setup Facebook DINOv2 processor & model...")
    # processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    # model = AutoModel.from_pretrained("facebook/dinov2-base")

    logger.info("Setting seeds for reproducibility...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Check for CUDA availability and set seeds accordingly
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        logger.warning("CUDA not available. Running on CPU.")

    logger.info("Setup Facebook DINOv2 processor & model...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    # Move model to CUDA if available, otherwise keep on CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

    logger.info("Loading Google news Word2Vec model...")
    # TODO: Find Better cache solution for this model
    model_txt = load_google_news_vectors()
    # Precompute and cache vector norms to speed up similarity/WMD calculations
    model_txt.fill_norms(force=True)

    logger.info("Ensuring output directory exists...")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_image_path):
        raise FileNotFoundError(
            f"Input image not found at {input_image_path}"
        )

    logger.info("Generating original edited image & embedding...")
    # Generate image and embedding
    orig_image_path, orig_image, orig_embedding = generate_image_and_embedding(
        input_image_path=input_image_path,
        prompt=prompt,
        output_dir=output_dir,
        model_name=model_name,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        seed=seed,
        processor=processor,
        model=model
    )

    logger.info("Generating text perturbations...")
    responses, perturbations = generate_perturbations(text=prompt)

    logger.info("Running pix2pix perturbation pipeline...")
    image_distances = run_pix2pix_pipeline(
        processor=processor,
        model=model,
        prompt=prompt,
        input_image_path=input_image_path,
        output_dir=output_dir,
        model_name=model_name,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        seed=seed,
        perturbations=perturbations,
        orig_embedding=orig_embedding
    )

    logger.info("Computing WMD scores between prompt & perturbations...")
    wmd_scores = compute_wmd_scores(model_txt, prompt, responses)

    # TODO: In the main code we find similarty with inversed mode
    logger.info("Normalizing similarity scores...")
    sims = normalize_similarities(
        distances=image_distances,
        mode=mode,
    )

    logger.info("Fitting regression model...")
    linear_model, coeffs, weights = fit_weighted_regression(
        perturbations, sims, wmd_scores
    )

    logger.info("Computing metrics...")
    metrics = compute_regression_metrics(
        linear_model=linear_model,
        coeffs=coeffs,
        weights=weights,
        similarities=sims,
        perturbations=perturbations,
    )

    if plot_heatmap:
        logger.info("Plotting heatmap...")
        words = prompt.split()
        plot_text_heatmap(words, coeffs, "Text Heatmap")

    return {
        "orig_image_path": orig_image_path,
        "reponses": responses,
        "wmd_scores": wmd_scores,
        "similarities": sims,
        "coefficients": coeffs,
        "weights": weights,
        "metrics": metrics
    }
