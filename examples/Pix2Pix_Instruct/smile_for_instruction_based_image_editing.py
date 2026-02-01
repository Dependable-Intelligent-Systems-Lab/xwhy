import os
import time
import json
import base64
import logging
import gdown
import zipfile
import pickle
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union, Literal

from dotenv import load_dotenv
import requests
import tiktoken

import torch
from transformers import AutoImageProcessor, AutoModel
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler
)
from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights,
)

from PIL import Image, ImageDraw, ImageFont, ImageOps

import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import transforms

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from scipy.stats import wasserstein_distance
from gensim.models import KeyedVectors
import gensim.downloader as api
from img2img_turbo import run_inference_paired
from google import genai
from google.genai import types
from openai import OpenAI


load_dotenv()


# --------------------------------------------------------------------------- #
# Configure logger
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
# Surrogate model types
# --------------------------------------------------------------------------- #

class SurrogateMethod(Enum):
    """Enumeration for supported surrogate model types."""
    # Linear Models
    GLM_OLS = "glm_ols"        # Global/Unweighted Ordinary Least Squares (OLS)
    GLM_RIDGE = "glm_ridge"    # Global/Unweighted Ridge Regression (L2)
    LIME = "lime_ols"          # Local Weighted OLS
    LIME_RIDGE = "lime_ridge"  # Local Weighted Ridge Regression (L2)
    BAYLIME = "baylime"        # Bayesian Weighted Ridge (L2)
    # Non-Linear Tree-Based Models
    RANDOMFOREST = "randomforest"
    GRADIENT_BOOSTING = "gradientboosting"
    XGBOOST = "xgboost"


# --------------------------------------------------------------------------- #
# Download I2EBench [Dataset](https://github.com/cocoshe/I2EBench)
# --------------------------------------------------------------------------- #

def download_i2ebench_dataset(
    url: str = "https://drive.google.com/uc?id=10X2C6INLqhY_hbgnOcUNvBD03P-cpX78",
    output_filename: str = "i2ebench.zip",
    extract_dir: str = "i2ebench"
) -> str:
    """
    Downloads the I2EBench dataset from Google Drive and extracts it.

    Args:
        url (str): Google Drive URL of the dataset zip file.
        output_filename (str): Local filename to save the downloaded zip file.
        extract_dir (str): Directory where the zip file contents will be extracted.

    Returns:
        str: The path to the extracted directory.
    """

    print(f"Downloading dataset from: {url} to {output_filename}")
    gdown.download(url, output_filename, quiet=False)

    print(f"Creating extraction directory: {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)

    print(f"Extracting {output_filename} to: {extract_dir}")
    with zipfile.ZipFile(output_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Dataset extracted to: {extract_dir}")
    return extract_dir


def load_i2ebench_data(
    root_dir: str = "i2ebench",
    categories_dir: List[str] | None = None,
    limits_per_category: Union[List[int], int] = 1
) -> Dict[str, List[Tuple[str, str]]]:
    """Parses the I2EBench dataset with limits and file validation.

    Args:
        root_dir: The root directory of the dataset. Defaults to "i2ebench".
        categories_dir: A list of category directory names to process.
            Defaults to the standard 8 categories if None.
        limits_per_category: Defines how many items to load for each category.
            - If a List[int]: Must match the length of categories_dir.
            - If an int (N): It sets the limit to N for ALL categories, resulting
              in a list like [N, N, N, ...] with length equal to categories_dir.

    Returns:
        A dictionary where keys are category names and values are lists of 
        tuples. Each tuple contains (full_image_path, prompt).

    Raises:
        FileNotFoundError: If main directories or JSON files are missing.
        ValueError: If the lengths of limits and categories do not match.
    """
    # Default mutable argument handling
    if categories_dir is None:
        categories_dir = [
            'Deblurring', 'HazeRemoval', 'Lowlight', 'NoiseRemoval', 
            'RainRemoval', 'ShadowRemoval', 'SnowRemoval', 'WatermarkRemoval'
        ]

    # 1. Validate root directory structure
    edit_data_path = os.path.join(root_dir, "EditBench", "EditData")
    if not os.path.exists(edit_data_path):
        raise FileNotFoundError(f"The directory '{edit_data_path}' does not exist.")

    # 2. Handle limit generation logic
    if isinstance(limits_per_category, int):
        # Repeat the single integer value for all categories.
        limit_value = limits_per_category
        limits_per_category = [limit_value] * len(categories_dir)

    # 3. Check length alignment
    if len(limits_per_category) != len(categories_dir):
        raise ValueError(
            f"Length mismatch: 'limits_per_category' has {len(limits_per_category)} elements, "
            f"but 'categories_dir' has {len(categories_dir)} elements."
        )

    # 4. Check existence of all category directories
    for category in categories_dir:
        category_path = os.path.join(edit_data_path, category)
        if not os.path.exists(category_path):
            raise FileNotFoundError(f"Category directory not found: {category_path}")

    # 5. Process data
    dataset_dict = {}

    for index, category in enumerate(categories_dir):
        limit = limits_per_category[index]

        category_path = os.path.join(edit_data_path, category)
        json_file_path = os.path.join(category_path, f"{category}.json")
        image_input_dir = os.path.join(category_path, "input")

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file missing: {json_file_path}")

        items_list = []

        for key, info in data.items():
            # STOP condition: Check if we successfully collected enough items
            if len(items_list) >= limit:
                break

            image_filename = info.get("image")
            prompt = info.get("ori_exp")

            if image_filename and prompt:
                full_img_path = os.path.join(image_input_dir, image_filename)

                # Validation: Check if the image file actually exists
                if not os.path.exists(full_img_path):
                    # Skip this item if the file is missing
                    continue

                items_list.append((full_img_path, prompt))

        dataset_dict[category] = items_list

    return dataset_dict


# --------------------------------------------------------------------------- #
# Save and load data to/from pickle
# --------------------------------------------------------------------------- #

def save_to_pickle(output_path: str = "pickle_data.pkl", **data: Any) -> None:
    """
    Saves keyword arguments directly to a pickle file.

    Args:
        output_path (str): Full path (including filename) where the
            output pickle should be saved.
        **data: Arbitrary named data to persist.
    """
    try:
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Data successfully saved to: {output_path}")
    except Exception as e:
        print(f"Error saving data to pickle file: {e}")
        raise


def load_from_pickle(file_path: str) -> Dict[str, Any]:
    """
    Loads and returns the object stored in a pickle file.

    This function returns exactly the same dictionary that was written
    by `save_to_pickle`, without modifying its structure or contents.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        Dict[str, Any]: The unpickled dictionary containing the data
        that was originally saved.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        Exception: If an error occurs during unpickling.
    """
    try:
        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)
        print(f"Data successfully loaded from: {file_path}")
        return loaded_data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading data from pickle file: {e}")
        raise


# --------------------------------------------------------------------------- #
# Word2Vec model
# --------------------------------------------------------------------------- #

def _download_file_fast(url: str, output_path: str) -> None:
    """
    Download a file quickly, using requests.
    Includes a basic check for file size to detect failed downloads.
    """
    logger.info(f"Attempting direct download from {url} to {output_path} using requests...")
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        # Minimum expected size for GoogleNews-vectors-negative300.bin.gz is ~1.5GB
        min_expected_compressed_size_bytes = 100 * 1024 * 1024 # 100 MB as a sanity check

        downloaded_size = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)

        if downloaded_size < min_expected_compressed_size_bytes:
            raise IOError(
                f"Downloaded file is too small ({downloaded_size / (1024*1024):.2f} MB). "
                f"Expected compressed file size > {min_expected_compressed_size_bytes / (1024*1024):.2f} MB."
            )
        logger.info(f"Successfully downloaded {downloaded_size / (1024*1024):.2f} MB to {output_path}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error during direct download from {url}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    except IOError as e:
        logger.error(f"File system or size check error: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise


def _extract_gzip(gz_path: str, output_path: str) -> None:
    """
    Extract a .gz file safely.
    """
    import gzip
    import shutil

    logger.info("Extracting gzip archive...")
    with gzip.open(gz_path, "rb") as f_in, open(output_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


# TODO: Improving and using better Word2Vec Models and also giving user an option
# to have preference on this

def load_google_news_vectors(
    cache_dir: str = "~/.cache/google_news_vectors",
    force_download: bool = False,
) -> KeyedVectors:
    """
    Load Google News Word2Vec vectors with local caching, prioritizing gensim.downloader.

    Args:
        cache_dir (str): Directory where the final .bin model will be stored/loaded from.
        force_download (bool): If True, bypass existing local cache and force re-download/re-load from gensim.
                               If gensim.downloader also bypasses its cache, it will re-download.

    Returns:
        KeyedVectors: Loaded Google News Word2Vec model.
    """
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    final_bin_path = os.path.join(cache_dir, "GoogleNews-vectors-negative300.bin")

    # 1. Try to load from the user-specified cache_dir first (unless force_download)
    if os.path.exists(final_bin_path) and not force_download:
        logger.info(f"Loading Google News vectors from specified local cache: {final_bin_path}...")
        return KeyedVectors.load_word2vec_format(final_bin_path, binary=True)

    # 2. If not in cache_dir or force_download, use gensim.downloader
    logger.info("Attempting to load 'word2vec-google-news-300' via gensim.downloader...")
    try:
        # gensim.downloader manages its own caching (usually in ~/.gensim/data)
        model = api.load("word2vec-google-news-300")
        logger.info("Successfully loaded model via gensim.downloader.")

        # Save the model to the user's specified cache_dir for consistent storage
        logger.info(f"Saving gensim-loaded model to specified cache: {final_bin_path}...")
        model.save_word2vec_format(final_bin_path, binary=True)
        return model

    except Exception as e:
        logger.warning(f"Gensim.downloader failed to load 'word2vec-google-news-300': {e}. Falling back to direct HTTP download.")

        # 3. Fallback to direct HTTP download if gensim.downloader fails
        # This URL is prone to changing/breaking. It's a last resort.
        manual_gz_path = final_bin_path + ".gz"
        manual_download_url = (
            "https://public.ukp.informatik.tu-darmstadt.de/reimers/wordembeddings/GoogleNews-vectors-negative300.bin.gz"
        )
        try:
            _download_file_fast(manual_download_url, manual_gz_path)
            _extract_gzip(manual_gz_path, final_bin_path)
            logger.info("Successfully loaded Google News vectors via direct HTTP download.")
            return KeyedVectors.load_word2vec_format(final_bin_path, binary=True)
        except Exception as manual_e:
            logger.error(f"Direct HTTP download and extraction failed: {manual_e}")
            raise RuntimeError("Failed to load Google News Word2Vec model from all sources.") from manual_e


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


def inject_text_at_position(
    original_text: str,
    position: Literal["start", "middle", "end"],
    default_index: int = 4,
    inject_text: Optional[str] = None
) -> str:
    """Injects text into a string at a specific position based on a selection.

    The function uses a provided 'inject_text' string if available; otherwise,
    it selects a word from a default list using 'default_index'.

    Args:
        original_text (str): The base prompt or text.
        position (Literal["start", "middle", "end"]): The placement target.
        default_index (int): Index to select from ['could', 'would', 'should',
            '###', 'please']. Defaults to 4 ("please").
        inject_text (Optional[str]): A custom string to inject. If provided,
            it overrides the default_index selection.

    Returns:
        str: The new text with the injected word/phrase.

    Raises:
        ValueError: If the position is invalid or default_index is out of range.
    """
    default_options = ['could', 'would', 'should', '###', 'please']

    # Selection logic
    if inject_text is not None:
        word_to_add = inject_text
    else:
        if 0 <= default_index < len(default_options):
            word_to_add = default_options[default_index]
        else:
            raise ValueError(f"default_index must be between 0 and {len(default_options)-1}")

    words = original_text.split()

    if position == "start":
        words.insert(0, word_to_add)
    elif position == "middle":
        # Calculates middle based on word count
        mid = len(words) // 2
        words.insert(mid, word_to_add)
    elif position == "end":
        words.append(word_to_add)
    else:
        raise ValueError("Position must be 'start', 'middle', or 'end'.")

    return " ".join(words)


# --------------------------------------------------------------------------- #
# Generate image and embedding
# --------------------------------------------------------------------------- #

def _create_placeholder_image(
    prompt: str,
    output_dir: str,
    filename_prefix: str = "gemini_generated",
    save: bool = True
) -> Union[str, Image.Image]:
    """Creates a black placeholder image with error text using Pillow.

    Helper function to maintain consistency with the original logic.
    """
    img_size = (600, 600)
    black_image = Image.new('RGB', img_size, color='black')
    draw = ImageDraw.Draw(black_image)

    text_to_draw = f"image wasn't generated because of\n\"{prompt}\"\ndoesn't mean"

    try:
        font = ImageFont.truetype("arialbd.ttf", 40)  # Adjusted size for 600x600
    except IOError:
        font = ImageFont.load_default()

    # Calculate text position to center it
    bbox = draw.textbbox((0, 0), text_to_draw, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (img_size[0] - text_width) / 2
    y = (img_size[1] - text_height) / 2

    draw.text((x, y), text_to_draw, fill="white", font=font, align="center")

    if save:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"{filename_prefix}_{timestamp}.png"
        gen_path = os.path.join(output_dir, filename)

        black_image.save(gen_path)
        return gen_path
    else:
        return black_image


def extract_image_embedding(
    image_path: str,
    processor: Any,
    model: Any
) -> np.ndarray:
    """
    Extract an image embedding using a DINOv2 model.

    Args:
        image (str): Input image.
        processor (Any): DINOv2 image processor.
        model (Any): DINOv2 model.

    Returns:
        np.ndarray: Extracted embedding vector.
    """
    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt")

    inputs['pixel_values'] = inputs['pixel_values'].to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    emb = outputs.last_hidden_state.mean(dim=1)
    emb = emb.squeeze().cpu().numpy()
    return emb


def generate_image_gemini(
    input_image_path: str,
    prompt: Union[str, List[str]],
    output_dir: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash-image",
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 40,
    max_output_tokens: int = 8192,
    stream: bool = True,
    seed: Optional[int] = None,
) -> Union[Tuple[bool, str, float], List[Tuple[bool, str, float]]]:
    """Generate an edited image using the Gemini API.

    Supports single (standard mode) or batch mode (list of prompts).

    Args:
        input_image_path (str): Path to the original image.
        prompt (str | list[str]): Text instruction(s) for image editing.
        output_dir (str): Directory to store generated images.
        api_key (str): Gemini API key.
        model_name (str): Name of the Gemini model.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
        top_k (int): Top-k sampling parameter.
        max_output_tokens (int): Maximum output tokens.
        stream (bool): Whether to use streaming generation.
        seed (int | None): Random seed for reproducibility.

    Returns:
        tuple[bool, str, float] | list[tuple[bool, str, float]]:
            - bool (gen_img_flag): True if image was generated by AI, False if placeholder was used.
            - str: Path to the generated image.
            - float: The estimated cost of generation.

    Raises:
        FileNotFoundError: If the input image is not found.
        TypeError: If prompt is not str or list[str].
    """
    # TODO: modify this function to accept only prompt not list of prompt
    def generate_single(single_prompt: str, image_data: bytes, mime_type: str, client: genai.Client) -> Tuple[str, bool, float]:
        # Define pricing for Gemini models
        input_cost_2_5_flash = 0.0005
        output_cost_2_5_flash = 0.002
        input_cost_3_pro = 0.001
        output_cost_3_pro = 0.004

        text_part = types.Part.from_text(text=single_prompt)
        image_part = types.Part.from_bytes(data=image_data, mime_type=mime_type)
        contents = [types.Content(role="user", parts=[image_part, text_part])]

        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            response_modalities=["image", "text"],
            response_mime_type="text/plain",
            seed=seed
        )

        generated_img = None
        gen_img_flag = True
        cost = 0.0  # Initialize cost

        # Calculate input_cost based on model_name
        if model_name == "gemini-2.5-flash-image":
            input_cost = input_cost_2_5_flash
        elif model_name == "gemini-3-pro-image-preview":
            input_cost = input_cost_3_pro
        else:
            input_cost = 0.0

        try:
            if stream:
                response_iter = client.models.generate_content_stream(
                    model=model_name, contents=contents, config=generate_content_config
                )
                for chunk in response_iter:
                    for part in chunk.parts:
                        if part.inline_data is not None:
                            generated_img = Image.open(BytesIO(part.inline_data.data))
                            mime_type = part.inline_data.mime_type
                            # Calculate output_cost and update total cost
                            if model_name == "gemini-2.5-flash-image":
                                output_cost = output_cost_2_5_flash
                            elif model_name == "gemini-3-pro-image-preview":
                                output_cost = output_cost_3_pro
                            else:
                                output_cost = 0.0
                            cost = input_cost + output_cost
                            break  # Stop after first successful response
            else:
                response = client.models.generate_content(
                    model=model_name, contents=contents, config=generate_content_config
                )
                for part in response.parts:
                    if part.inline_data is not None:
                        generated_img = Image.open(BytesIO(part.inline_data.data))
                        mime_type = part.inline_data.mime_type
                        # Calculate output_cost and update total cost
                        if model_name == "gemini-2.5-flash-image":
                            output_cost = output_cost_2_5_flash
                        elif model_name == "gemini-3-pro-image-preview":
                            output_cost = output_cost_3_pro
                        else:
                            output_cost = 0.0
                        cost = input_cost + output_cost
                        break
        except Exception as e:
            logger.exception(f"Error during API call: {e}")

        if generated_img is None:
            gen_img_flag = False
            logger.info(f"Failed to generate image for prompt: '{single_prompt}'. Creating placeholder.")

            generated_img = _create_placeholder_image(
                prompt=single_prompt,
                output_dir=output_dir,
                save=False
            )
            cost = 0.0  # No cost for placeholder
            mime_type = "image/png"  # Force PNG for the placeholder

        # Determine file extension based on MIME type
        if mime_type == "image/jpeg":
            ext = ".jpg"
        elif mime_type == "image/png":
            ext = ".png"
        else:
            ext = ".png"

        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"gemini_generated_{timestamp}{ext}"
        gen_path = os.path.join(output_dir, filename)

        generated_img.save(gen_path)

        print(f"------------------- \"{gen_path}\" generated! (Success: {gen_img_flag}) -------------------")
        return gen_img_flag, gen_path, cost

    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    with open(input_image_path, "rb") as image_file:
        image_data = image_file.read()

    # Determine initial file extension based on path (for the request)
    if input_image_path.lower().endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    elif input_image_path.lower().endswith(".png"):
        mime_type = "image/png"
    else:
        mime_type = "image/jpeg"

    with genai.Client(api_key=api_key) as client:
        if isinstance(prompt, str):
            return generate_single(prompt, image_data, mime_type, client)
        elif isinstance(prompt, list):
            return [generate_single(p, image_data, mime_type, client) for p in prompt]
        else:
            raise TypeError("Prompt must be str or list[str] for batch mode.")


def generate_image_openai(
    input_image_path: str,
    prompt: Union[str, List[str]],
    output_dir: str,
    api_key: str,
    model_name: str = "gpt-image-1",
    size: str = "1024x1024",
    quality: str = "auto",
    n: int = 1,
) -> Union[Tuple[bool, str, float], List[Tuple[bool, str, float]]]:
    """
    Edit an image using the OpenAI Image Editing API (image + prompt).

    This behaves like Gemini image editing: the input image is modified
    according to the provided text instruction.

    Args:
        input_image_path (str): Path to the original image.
        prompt (str | list[str]): Text instruction(s) for image editing.
        output_dir (str): Directory to store generated images.
        api_key (str): OpenAI API key.
        model_name (str): Image model (default: gpt-image-1).
        size (str): Output image size (e.g., "1024x1024").
        quality (str): Image quality ("low", "medium", "high", and "auto").
        n (int): Number of images to generate (OpenAI usually supports 1).

    Returns:
        tuple[bool, str, float] | list[tuple[bool, str, float]]:
            - bool: True if generated by AI, False if placeholder was used
            - str: Path to the generated image
            - float: The estimated cost of generation.
    """

    if not api_key:
        raise ValueError("OpenAI API key is required.")

    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    os.makedirs(output_dir, exist_ok=True)

    client = OpenAI(api_key=api_key)

    def generate_single(single_prompt: str) -> Tuple[bool, str, float]:
        gen_img_flag = True
        gen_path = ""
        cost = 0.0  # Initialize cost

        # Define pricing for OpenAI models (per 1M tokens)
        INPUT_COST_PER_TOKEN_GPT_IMAGE_1 = 10.00 / 1_000_000
        OUTPUT_COST_PER_TOKEN_GPT_IMAGE_1 = 40.00 / 1_000_000
        INPUT_COST_PER_TOKEN_GPT_IMAGE_1_MINI = 2.50 / 1_000_000
        OUTPUT_COST_PER_TOKEN_GPT_IMAGE_1_MINI = 8.00 / 1_000_000

        # Output image token consumption based on quality and size
        output_token_consumption_map = {
            ("1024x1024", "low"): 272,
            ("1024x1024", "medium"): 1056,
            ("1024x1024", "high"): 4160,
            ("1024x1536", "low"): 408,
            ("1024x1536", "medium"): 1584,
            ("1024x1536", "high"): 6240,
            ("1536x1024", "low"): 400,
            ("1536x1024", "medium"): 1568,
            ("1536x1024", "high"): 6208,
        }

        try:
            # Calculate input prompt tokens using tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4") # Using gpt-4 tokenizer as a general tokenizer
            input_tokens = len(encoding.encode(single_prompt))

            # Determine input and output token costs per token based on model_name
            if "gpt-image-1" in model_name.lower():
                input_cost_per_token = INPUT_COST_PER_TOKEN_GPT_IMAGE_1
                output_cost_per_token = OUTPUT_COST_PER_TOKEN_GPT_IMAGE_1
            elif "gpt-image-1-mini" in model_name.lower():
                input_cost_per_token = INPUT_COST_PER_TOKEN_GPT_IMAGE_1_MINI
                output_cost_per_token = OUTPUT_COST_PER_TOKEN_GPT_IMAGE_1_MINI
            else:
                input_cost_per_token = 0.0
                output_cost_per_token = 0.0
                logger.warning(f"Unknown OpenAI model '{model_name}'. Cost will be 0.")

            # Calculate input prompt cost
            cost += input_tokens * input_cost_per_token

            with open(input_image_path, "rb") as img_file:
                response = client.images.edit(
                    model=model_name,
                    image=img_file,
                    prompt=single_prompt,
                    size=size,
                    quality=quality,
                    n=n,
                )

            if response.data and response.data[0].b64_json:
                image_bytes = BytesIO(
                    base64.b64decode(response.data[0].b64_json)
                )
                image_pil = Image.open(image_bytes)

                timestamp = int(time.time() * 1000)
                filename = f"openai_edited_{timestamp}.png"
                gen_path = os.path.join(output_dir, filename)
                image_pil.save(gen_path)

                # Calculate output image token cost based on quality and size
                # Default to medium quality token cost if 'auto' or not found
                output_tokens_per_image = output_token_consumption_map.get(
                    (size, quality.lower()), output_token_consumption_map.get(("1024x1024", "medium"))
                )
                if output_tokens_per_image is None:
                    logger.warning(f"Could not determine output token count for size '{size}' and quality '{quality}'. Using default medium (1024x1024) tokens.")
                    output_tokens_per_image = output_token_consumption_map[('1024x1024', 'medium')]

                cost += output_tokens_per_image * output_cost_per_token

                logger.info(
                    f"------------------- \"{gen_path}\" generated! "
                    f"(Success: {gen_img_flag}) -------------------"
                )
            else:
                raise RuntimeError("No image data returned from OpenAI.")

        except Exception as e:
            gen_img_flag = False
            cost = 0.0  # No cost for failed generation or placeholder
            logger.exception(
                f"Error editing image with OpenAI for prompt '{single_prompt}': {e}"
            )
            gen_path = _create_placeholder_image(
                prompt=single_prompt,
                output_dir=output_dir,
                filename_prefix="openai_generated"
            )

        return gen_img_flag, gen_path, cost

    if isinstance(prompt, str):
        return generate_single(prompt)
    elif isinstance(prompt, list):
        return [generate_single(p) for p in prompt]
    else:
        raise TypeError("Prompt must be str or list[str].")


def generate_image_seedream(
    input_image_path: str,
    prompt: Union[str, List[str]],
    output_dir: str,
    api_key: str, # This should be ARK_API_KEY from the example
    model_name: str = "seedream-4-5-251128", # Default model from user's example
    base_url: str = "https://ark.ap-southeast.bytepluses.com/api/v3", # Default base_url from user's example
    size: str = "2K", # Default size from user's example
    watermark: bool = False, # Default watermark from user's example
) -> Union[Tuple[bool, str, float], List[Tuple[bool, str, float]]]:
    """
    Generate an image using the BytePlus SeeDream API via the OpenAI client,
    following the image-to-image pattern provided in the example.

    Args:
        input_image_path (str): Path to the original image.
        prompt (str | list[str]): Text instruction(s) for image generation/editing.
        output_dir (str): Directory to store generated images.
        api_key (str): BytePlus ARK API key.
        model_name (str): SeeDream model name (e.g., \"seedream-4-5-251128\").
        base_url (str): Base URL for the BytePlus ARK API.
        size (str): Output image size (e.g., \"2K\").
        watermark (bool): Whether to add a watermark (False by default).

    Returns:
        tuple[bool, str, float] | list[tuple[bool, str, float]]:
            - bool: True if generated by AI, False if placeholder was used
            - str: Path to the generated image
            - float: The estimated cost of generation.

    Raises:
        ValueError: If BytePlus ARK API key is missing.
        FileNotFoundError: If the input image is not found.
        TypeError: If prompt is not str or list[str].
    """

    if not api_key:
        raise ValueError("BytePlus ARK API key is required.")

    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Initialize OpenAI client with BytePlus ARK endpoint and API key
    client = OpenAI(base_url=base_url, api_key=api_key)

    # Read input image and convert to base64 data URI
    with open(input_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        mime_type = "image/jpeg"
        if input_image_path.lower().endswith(".png"):
            mime_type = "image/png"
        input_image_data_uri = f"data:{mime_type};base64,{encoded_string}"

    def _generate_single_see_dream(single_prompt: str) -> Tuple[bool, str, float]:
        gen_img_flag = True
        gen_path = ""
        cost = 0.0  # Initialize cost

        # Define pricing for SeeDream models per image
        price_seedream_4_5 = 0.04
        price_seedream_4_0 = 0.03

        try:
            images_response = client.images.generate(
                model=model_name,
                prompt=single_prompt,
                size=size,
                response_format="url", # The example shows 'url' as response_format
                extra_body={
                    "image": input_image_data_uri, # Input image as base64 data URI
                    "watermark": watermark,
                }
            )

            if images_response.data and images_response.data[0].url:
                image_url = images_response.data[0].url
                logger.info(f"BytePlus SeeDream generated image URL: {image_url}")

                # Download the image from the URL
                response = requests.get(image_url)
                response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

                image_bytes = BytesIO(response.content)
                image_pil = Image.open(image_bytes)

                timestamp = int(time.time() * 1000)
                filename = f"seedream_generated_{timestamp}.png"
                gen_path = os.path.join(output_dir, filename)
                image_pil.save(gen_path)

                # Assign cost based on model name for successful generation
                if model_name == "seedream-4-5-251128":
                    cost = price_seedream_4_5
                elif model_name == "seedream-4-0-250828":
                    cost = price_seedream_4_0
                else:
                    logger.warning(f"Unknown SeeDream model '{model_name}'. Cost will be 0.")

                logger.info(
                    f"------------------- \"{gen_path}\" generated by SeeDream! "
                    f"(Success: {gen_img_flag}) -------------------"
                )
            else:
                raise RuntimeError("No image URL returned from BytePlus SeeDream API response.")

        except Exception as e:
            gen_img_flag = False
            cost = 0.0  # No cost for failed generation or placeholder
            logger.exception(
                f"Error generating image with BytePlus SeeDream for prompt '{single_prompt}': {e}"
            )
            # Use the placeholder image function if generation fails
            gen_path = _create_placeholder_image(
                prompt=single_prompt,
                output_dir=output_dir,
                filename_prefix="seedream_generated"
            )

        return gen_img_flag, gen_path, cost

    if isinstance(prompt, str):
        return _generate_single_see_dream(prompt)
    elif isinstance(prompt, list):
        return [_generate_single_see_dream(p) for p in prompt]
    else:
        raise TypeError("Prompt must be str or list[str].")


def _get_deeplab_mask(
    image: Image.Image,
    model: Any,
    weights: Any
) -> Image.Image:
    """
    Helper function to generate a binary mask using a pre-loaded DeepLabV3 model.
    Masks the foreground (objects) as White (255) and background as Black (0).
    """
    try:
        # Move model to device if not already there
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Check if model is already on the correct device to avoid overhead
        if next(model.parameters()).device.type != device:
            model.to(device)

        model.eval()

        # Preprocess using the provided weights transforms
        preprocess = weights.transforms()
        input_tensor = preprocess(image).unsqueeze(0)

        input_tensor = input_tensor.to(device)

        # Inference
        with torch.no_grad():
            output = model(input_tensor)['out'][0]

        # Output is (21, H, W). 0 is background.
        output_predictions = output.argmax(0)

        # Create mask: 1 where class != 0 (background), else 0
        mask_arr = (output_predictions != 0).byte().cpu().numpy() * 255

        # Convert to PIL
        mask_image = Image.fromarray(mask_arr).convert("L")

        # Resize mask back to original image size if transforms changed it
        if mask_image.size != image.size:
            mask_image = mask_image.resize(image.size, resample=Image.NEAREST)

        return mask_image
    except Exception as e:
        logger.error(f"Failed to generate mask with DeepLabV3: {e}")
        # Return a blank white mask (edit everything) as fallback
        return Image.new("L", image.size, 255)


def generate_image_huggingface(
    input_image_path: str,
    prompt: str,
    pipe: Any,
    output_dir: str = "outputs",
    mask_model: Optional[Any] = None,
    mask_model_weights: Optional[Any] = None
) -> Tuple[bool, str, float]:
    """
    Generates an edited image using a HuggingFace Diffusers pipeline.

    Args:
        input_image_path (str): Path to the original input image.
        prompt (str): Text instruction describing how the input image should be modified.
        pipe (Any): Pre-initialized HuggingFace Diffusers pipeline object (e.g., StableDiffusionInstructPix2PixPipeline).
        output_dir (str): Directory where the generated image will be saved.
        mask_model (Optional[Any]): An optional model that takes a PIL Image and returns a PIL Image mask.
        mask_model_weights (Optional[Any]): Weights/Transforms associated with the mask_model.

    Returns:
        Tuple[bool, str, float]: A tuple containing:
            - gen_img_flag (bool): True if the image was successfully generated, False otherwise.
            - gen_path (str): Path to the generated image or placeholder image.
            - cost (float): The estimated cost of generation (0.0 for local models).
    """
    gen_img_flag = True
    gen_path = ""
    cost = 0.0  # Initialize cost for local models

    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    os.makedirs(output_dir, exist_ok=True)

    image = Image.open(input_image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    mask_image = None
    if mask_model is not None:
        mask_image = _get_deeplab_mask(image, mask_model, mask_model_weights)

    try:
        if isinstance(pipe, StableDiffusionInpaintPipeline):
            if mask_model is None or mask_model_weights is None:
                raise ValueError("mask_model and mask_model_weights are required for Inpainting pipelines.")
            images = pipe(
                prompt=prompt,
                image=image,
                mask_image=mask_image,
                num_inference_steps=50,
            ).images
        elif isinstance(pipe, StableDiffusionInstructPix2PixPipeline):
            images = pipe(
                prompt,
                image=image,
                num_inference_steps=10,
                image_guidance_scale=1
            ).images
        else:
            raise ValueError(f"Unsupported HuggingFace pipeline type: {type(pipe)}")

        image_pil = images[0]

        timestamp = int(time.time() * 1000)
        filename = f"huggingface_generated_{timestamp}.png"
        gen_path = os.path.join(output_dir, filename)
        image_pil.save(gen_path)
        logger.info(
            f"------------------- \"{gen_path}\" generated! "
            f"(Success: {gen_img_flag}) -------------------"
        )
    except Exception as e:
        gen_img_flag = False
        cost = 0.0  # No cost for failed generation or placeholder
        logger.exception(
            f"Error generating image with HuggingFace pipeline for prompt '{prompt}': {e}"
        )
        # Use the placeholder image function if generation fails
        gen_path = _create_placeholder_image(
            prompt=prompt,
            output_dir=output_dir,
            filename_prefix="huggingface_generated"
        )

    return gen_img_flag, gen_path, cost


def _generate_single_image(
    input_image_path: str,
    prompt: str,
    model_name: str,
    output_dir: str = "outputs",
    low_threshold: Optional[int] = None,
    high_threshold: Optional[int] = None,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    openai_size: str = "1024x1024",
    openai_quality: str = "auto",
    openai_n: int = 1,
    seedream_size: str = "2K",
    seedream_watermark: bool = False,
    huggingface_pipe: Optional[Any] = None,
    mask_model: Optional[Any] = None,
    mask_model_weights: Optional[Any] = None
) -> Tuple[bool, str, float]:
    """Generate a single edited image using a specified image model.

    This function dispatches the image generation request based on the
    selected model:

    - **Gemini models**: Use remote image editing with image + prompt.
    - **OpenAI models**: Use OpenAI image editing (image + prompt).
    - **BytePlus SeeDream models**: Use BytePlus SeeDream image editing (image + prompt).
    - **HuggingFace InstructPix2Pix**: Uses a local pre-initialized pipeline.
    - **Local models**: Use paired inference (e.g., Pix2Pix, ControlNet).

    Args:
        input_image_path (str):
            Path to the original input image used as the editing source.
        prompt (str):
            Text instruction describing how the input image should be
            modified.
        model_name (str):
            Name of the image generation or editing model. This value
            determines which backend (Gemini, OpenAI, or local) is used.
        output_dir (str, optional):
            Directory where the generated image will be saved.
            Defaults to ``"outputs"``.
        low_threshold (Optional[int]):
            Lower Canny edge threshold used by local paired-inference
            models. Ignored for Gemini and OpenAI models.
        high_threshold (Optional[int]):
            Upper Canny edge threshold used by local paired-inference
            models. Ignored for Gemini and OpenAI models.
        seed (Optional[int]):
            Random seed for reproducibility. Passed through to all
            backends that support deterministic generation.
        api_key (Optional[str]):
            API key required for remote models (Gemini or OpenAI).
            Must be provided when using those backends.
        openai_size (str):
            Output image resolution for OpenAI image editing
            (e.g., ``"1024x1024"``, ``"512x512"``).
        openai_quality (str):
            Quality setting for OpenAI image generation
            (e.g., ``"low"``, ``"medium"``).
        openai_n (int):
            Number of images to generate using OpenAI models.
            Most OpenAI image-editing models currently support only ``1``.
        seedream_size (str):
            Output image resolution for OpenAI image editing
            (e.g., ``"2K"``).
        seedream_watermark (bool):
            Whether to add a watermark to images generated by SeeDream models.
        huggingface_pipe (Optional[Any]): Pre-initialized HuggingFace pipe for
            models like `instruct-pix2pix`.
        mask_model (Optional[Any]): An optional model that takes a PIL Image
            and returns a PIL Image mask.
        mask_model_weights (Optional[Any]): Weights/Transforms associated with
            the mask_model.

    Returns:
        Tuple[bool, str, float]: # Updated return type
            A tuple containing:
            - **bool**: Indicates whether the image was successfully
              generated by the model (``True``)
              or a placeholder image was used (``False``).
            - **str**: File path to the generated (or placeholder) image.
            - **float**: The estimated cost of generation.

    Raises:
        ValueError:
            If an API key is required but not provided.
        FileNotFoundError:
            If no output image is produced by a local model.
        RuntimeError:
            If an unexpected error occurs during image generation.
    """
    model_name_lc = model_name.lower()

    # --------------------------------------------------
    # Gemini (image + prompt)
    # --------------------------------------------------
    if "gemini" in model_name_lc:
        if api_key is None:
            raise ValueError("API key is required for Gemini models.")

        gen_img_flag, gen_path, cost = generate_image_gemini(
            input_image_path=input_image_path,
            prompt=prompt,
            output_dir=output_dir,
            api_key=api_key,
            model_name=model_name,
            seed=seed,
        )
        return gen_img_flag, gen_path, cost

    # --------------------------------------------------
    # OpenAI (image + prompt editing)
    # --------------------------------------------------
    if any(key in model_name_lc for key in ("openai", "dall-e", "gpt-image")):
        if api_key is None:
            raise ValueError("API key is required for OpenAI models.")

        gen_img_flag, gen_path, cost = generate_image_openai(
            input_image_path=input_image_path,
            prompt=prompt,
            output_dir=output_dir,
            api_key=api_key,
            model_name=model_name,
            size=openai_size,
            quality=openai_quality,
            n=openai_n,
        )
        return gen_img_flag, gen_path, cost

    # --------------------------------------------------
    # BytePlus SeeDream (image + prompt editing)
    # --------------------------------------------------
    if "seedream" in model_name_lc:
        if api_key is None:
            raise ValueError("API key is required for BytePlus SeeDream models.")

        # Validate specific SeeDream models as requested
        if model_name not in ["seedream-4-5-251128", "seedream-4-0-250828"]:
            raise ValueError(
                f"Unsupported SeeDream model: {model_name}. "
                "Please use 'seedream-4-5-251128' or 'seedream-4-0-250828'."
            )

        gen_img_flag, gen_path, cost = generate_image_seedream(
            input_image_path=input_image_path,
            prompt=prompt,
            output_dir=output_dir,
            api_key=api_key,
            model_name=model_name,
            size=seedream_size,
            watermark=seedream_watermark,
        )
        return gen_img_flag, gen_path, cost

    # --------------------------------------------------
    # HuggingFace models (local pipeline)
    # --------------------------------------------------
    if any(key in model_name_lc for key in ("instruct-pix2pix", "stable-diffusion-2-inpainting")) and huggingface_pipe is not None:
        gen_img_flag, gen_path, cost = generate_image_huggingface(
            input_image_path=input_image_path,
            prompt=prompt,
            pipe=huggingface_pipe,
            output_dir=output_dir,
            mask_model=mask_model,
            mask_model_weights=mask_model_weights,
        )
        return gen_img_flag, gen_path, cost

    # --------------------------------------------------
    # Local paired inference (fallback for other local models)
    # --------------------------------------------------
    try:
        gen_img_flag = True
        cost = 0.0  # Initialize cost for local models
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
            os.path.join(output_dir, fname)
            for fname in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, fname))
        ]

        if not files:
            raise FileNotFoundError(
                f"No files found in directory: {output_dir}"
            )

        generated_file = max(files, key=os.path.getctime)
        path = Path(generated_file)
        timestamp = int(time.time() * 1000)

        gen_path = str(path.rename(
            path.with_name(f"img2img-turbo_generated_{timestamp}{path.suffix}")
        ))

        return gen_img_flag, gen_path, cost

    except FileNotFoundError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Unexpected error during image generation: {exc}"
        ) from exc


def submit_gemini_batch_job(
    api_key: str,
    image_path: str,
    text_list: List[str],
    model_name: str = "gemini-2.5-flash-image",
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 40,
    max_output_tokens: int = 8192,
    seed: Optional[int] = None,
    response_mime_type: str = "text/plain"
) -> str:
    """Submits a batch image generation job to the Gemini API.

    Args:
        api_key (str): Gemini API Key.
        image_path (str): Path to the source image.
        text_list (List[str]): List of text prompts.
        model_name (str): Model name.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
        top_k (int): Top-k sampling parameter.
        max_output_tokens (int): Maximum output tokens.
        seed (int | None): Random seed for reproducibility.
        response_mime_type (str): MIME type for the response.

    Returns:
        str: The unique resource name of the submitted batch job.
    """
    # TODO: use context manager for the client
    client = genai.Client(api_key=api_key)

    logger.info(f"Uploading image file: {image_path}")
    image_file = client.files.upload(file=image_path)
    logger.info(f"Uploaded image file: {image_file.name} (MIME: {image_file.mime_type})")

    # Define list of requests
    requests_data = []

    for ix, text in enumerate(text_list):
        custom_id = f"request_{ix}_image"

        # Build generation_config dictionary dynamically
        gen_config_dict = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": response_mime_type
        }
        if seed is not None:
            gen_config_dict["seed"] = seed

        requests_data.append({
            "custom_id": custom_id,
            "request": {
                "contents": [
                    {
                        "parts": [
                            {"text": text},
                            {
                                "file_data": {
                                    "file_uri": image_file.uri,
                                    "mime_type": image_file.mime_type
                                }
                            }
                        ]
                    }
                ],
                "generation_config": gen_config_dict
            }
        })

    json_file_path = 'batch_image_gen_requests.json'

    logger.info(f"\nCreating JSONL file: {json_file_path}")
    with open(json_file_path, 'w') as f:
        for req in requests_data:
            f.write(json.dumps(req) + '\n')

    logger.info(f"Uploading JSONL file: {json_file_path}")
    batch_input_file = client.files.upload(file=json_file_path)
    logger.info(f"Uploaded JSONL file: {batch_input_file.name}")

    logger.info("\nCreating batch job...")
    batch_multimodal_job = client.batches.create(
        model=model_name,
        src=batch_input_file.name,
        config={
            'display_name': 'my-batch-image-gen-job',
        }
    )
    logger.info(f"Created batch job: {batch_multimodal_job.name}")

    return batch_multimodal_job.name


def retrieve_gemini_batch_results(
    api_key: str,
    job_name: str,
    text_list: List[str],
    model_name: str,
    output_dir: str = "outputs"
) -> Tuple[List[Tuple[bool, str]], float]:
    """Polls for completion of a Gemini batch job and saves the results.

    Args:
        api_key (str): Gemini API Key.
        job_name (str): The unique resource name of the batch job to check.
        text_list (List[str]): Original list of prompts (needed for order/fallbacks).
        model_name (str): The Gemini model name used for batch generation.
        output_dir (str): Directory to save outputs.

    Returns:
        Tuple[List[Tuple[bool, str]], float]: A tuple containing:
            - List[Tuple[bool, str]]: A list of tuples (generation_flag, image_path).
                - generation_flag: True if generated by AI, False if placeholder.
                - image_path: Path to the saved image file.
            - float: The total estimated cost for the entire batch generation.
    """
    # TODO: use context manager for the client
    client = genai.Client(api_key=api_key)

    logger.info(f"Polling status for job: {job_name}")
    logger.info("Waiting for job to complete (this may take a moment)...")

    # Define batch pricing (half of standard Gemini pricing)
    batch_input_cost_2_5_flash = 0.0005 / 2
    batch_output_cost_2_5_flash = 0.002 / 2
    batch_input_cost_3_pro = 0.001 / 2
    batch_output_cost_3_pro = 0.004 / 2

    # Polling loop to wait for the job to finish
    while True:
        batch_multimodal_job = client.batches.get(name=job_name)
        state = batch_multimodal_job.state.name

        if state in ['JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED']:
            logger.info(f"\nJob finished with state: {state}")
            break

        time.sleep(30)  # Wait 30 seconds before checking again
        print(".", end="", flush=True)

    # Dictionary to store results: {custom_id: (flag, path)}
    processed_results = {}
    total_cost = 0.0
    os.makedirs(output_dir, exist_ok=True)

    if batch_multimodal_job.state.name == 'JOB_STATE_SUCCEEDED':
        result_file_name = batch_multimodal_job.dest.file_name
        logger.info(f"Results available in file: {result_file_name}")

        logger.info("Downloading and parsing result file content...")
        file_content_bytes = client.files.download(file=result_file_name)
        file_content = file_content_bytes.decode('utf-8')

        # Determine the cost per item for the specified model in batch mode
        item_input_cost = 0.0
        item_output_cost = 0.0
        if model_name == "gemini-2.5-flash-image":
            item_input_cost = batch_input_cost_2_5_flash
            item_output_cost = batch_output_cost_2_5_flash
        elif model_name == "gemini-3-pro-image-preview":
            item_input_cost = batch_input_cost_3_pro
            item_output_cost = batch_output_cost_3_pro
        else:
            logger.warning(f"Unknown Gemini model '{model_name}'. Cost will be 0.")

        # Parse each line of the JSONL result
        for line in file_content.splitlines():
            if not line:
                continue

            # The result file is also a JSONL file. Parse each line.
            try:
                parsed_response = json.loads(line)
                custom_id = parsed_response.get('custom_id') or parsed_response.get('key')
                current_item_cost = 0.0

                # Check for successful response structure
                if (
                    'response' in parsed_response
                    and 'candidates' in parsed_response['response']
                    and parsed_response['response']['candidates']
                ):
                    candidates = parsed_response['response']['candidates'][0]
                    found_image = False

                    if 'content' in candidates and 'parts' in candidates['content']:
                        for part in candidates['content']['parts']:
                            if 'inlineData' in part:
                                # Found an image!
                                mime = part['inlineData']['mimeType']
                                data = base64.b64decode(part['inlineData']['data'])

                                # Determine extension
                                ext = ".png" if "png" in mime else ".jpg"
                                timestamp = int(time.time() * 1000)
                                filename = f"{custom_id}_{timestamp}{ext}"
                                save_path = os.path.join(output_dir, filename)

                                # Save Image
                                with open(save_path, "wb") as img_f:
                                    img_f.write(data)

                                current_item_cost = item_input_cost + item_output_cost
                                processed_results[custom_id] = (True, save_path)
                                found_image = True
                                total_cost += current_item_cost  # Accumulate total cost
                                break

                    if not found_image:
                        # API returned a candidate but no inlineData (likely text refusal or filter)
                        logger.warning(f"No image found in response for {custom_id}")
                        processed_results[custom_id] = (False, None)  # Placeholder, no path yet
                else:
                    logger.warning(f"Invalid response structure for {custom_id}")
                    processed_results[custom_id] = (False, None)  # Placeholder, no path yet

            except Exception as e:
                logger.error(f"Error parsing line: {e}")
                # If parsing fails, treat as failed generation with no cost
                custom_id_from_error = line.split('"custom_id": "')[1].split('"')[0] if '"custom_id":' in line else f"unknown_error_{time.time()}"
                processed_results[custom_id_from_error] = (False, None)
    else:
        logger.warning(f"Job failed or was cancelled. Final state: {batch_multimodal_job.state.name}")

    # Final Compilation: Ensure return list matches input text_list order
    final_output_list = []

    for ix, text in enumerate(text_list):
        custom_id = f"request_{ix}_image"

        if custom_id in processed_results:
            flag, path = processed_results[custom_id]
            if path is None:  # If path wasn't set (e.g., API refusal),
                placeholder_path = _create_placeholder_image(
                    prompt=text,
                    output_dir=output_dir,
                    filename_prefix=custom_id
                )
                final_output_list.append((False, placeholder_path))
            else:
                final_output_list.append((flag, path))
        else:
            logger.warning(f"Generating placeholder for missing result: {custom_id}")
            placeholder_path = _create_placeholder_image(
                prompt=text,
                output_dir=output_dir,
                filename_prefix=custom_id
            )
            final_output_list.append((False, placeholder_path))

    return final_output_list, total_cost


# --------------------------------------------------------------------------- #
# Generate image from prompts and calculate wasserstein distance
# --------------------------------------------------------------------------- #

def generate_images_from_prompts(
    input_image_path: str,
    prompts: List[str],
    model_name: str,
    output_dir: str = "outputs",
    low_threshold: Optional[int] = None,
    high_threshold: Optional[int] = None,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    openai_size: str = "1024x1024",
    openai_quality: str = "auto",
    openai_n: int = 1,
    seedream_size: str = "2K",
    seedream_watermark: bool = False,
    huggingface_pipe: Optional[Any] = None,
    mask_model: Optional[Any] = None,
    mask_model_weights: Optional[Any] = None
) -> Tuple[List[Tuple[bool, str]], float]:
    """Generate multiple images from a list of text prompts.

    Args:
        input_image_path (str): Path to the original input image.
        prompts (List[str]): List of text prompts.
        model_name (str): Name of the image generation model.
        output_dir (Optional[str]): Directory for saving outputs.
        low_threshold (Optional[int]): Canny low threshold value.
        high_threshold (Optional[int]): Canny high threshold value.
        seed (Optional[int]): Random seed for reproducibility.
        api_key (Optional[str]): API key for remote models.
        openai_size (str):
            Output image resolution for OpenAI image editing
            (e.g., ``"1024x1024"``, ``"512x512"``).
        openai_quality (str):
            Quality setting for OpenAI image generation
            (e.g., ``"low"``, ``"medium"``).
        openai_n (int):
            Number of images to generate using OpenAI models.
            Most OpenAI image-editing models currently support only ``1``.
        seedream_size (str):
            Output image resolution for OpenAI image editing
            (e.g., ``"2K"``).
        seedream_watermark (bool):
            Whether to add a watermark to images generated by SeeDream models.
        huggingface_pipe (Optional[Any]): Pre-initialized HuggingFace pipe for
            models like `instruct-pix2pix`.
        mask_model (Optional[Any]): An optional model that takes a PIL Image
            and returns a PIL Image mask.
        mask_model_weights (Optional[Any]): Weights/Transforms associated with
            the mask_model.

    Returns:
        Tuple[List[Tuple[bool, str]], float]: A tuple containing:
            - List[Tuple[bool, str]]: Generation flags and image paths.
            - float: The total estimated cost of all image generations.
    """
    total_prompts = len(prompts)
    generated_images_paths = []
    total_cost = 0.0  # Initialize total cost

    logger.info(
        f"Starting image generation: {total_prompts} prompts | model={model_name}"
    )

    start_time = time.perf_counter()

    for idx, text in enumerate(prompts, start=1):
        iter_start = time.perf_counter()

        logger.debug(f"Prompt {idx} text: {text}")

        gen_img_flag, gen_path, cost = _generate_single_image(
            input_image_path=input_image_path,
            prompt=text,
            output_dir=output_dir,
            model_name=model_name,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            seed=seed,
            api_key=api_key,
            openai_size=openai_size,
            openai_quality=openai_quality,
            openai_n=openai_n,
            seedream_size=seedream_size,
            seedream_watermark=seedream_watermark,
            huggingface_pipe=huggingface_pipe,
            mask_model=mask_model,
            mask_model_weights=mask_model_weights,
        )

        generated_images_paths.append((gen_img_flag, gen_path))
        total_cost += cost

        iter_duration = time.perf_counter() - iter_start
        elapsed = time.perf_counter() - start_time
        avg_time = elapsed / idx
        eta = avg_time * (total_prompts - idx)

        # Throttled logging for large batches
        if idx % 5 == 0 or idx == total_prompts:
            logger.info(
                f"Progress {idx}/{total_prompts} | "
                f"Success={gen_img_flag} | "
                f"Iter={iter_duration:.4f}s | "
                f"ETA={eta:.4f}s | "
                f"Cost={cost:.4f} | "
                f"Total={total_cost:.4f}"
            )

        if not gen_img_flag:
            logger.warning(
                f"Image generation failed at {idx}/{total_prompts} "
                f"(path={gen_path})"
            )

    total_duration = time.perf_counter() - start_time
    avg_duration = total_duration / max(total_prompts, 1)

    logger.info(
        f"Image generation completed: {total_prompts} prompts | "
        f"Total cost={total_cost:.4f} | "
        f"Total time={total_duration:.4f}s | "
        f"Avg/prompt={avg_duration:.4f}s"
    )

    return generated_images_paths, total_cost


def compute_wasserstein_distances(
    input_image_path: str,
    generated_images: List[Tuple[bool, str]],
    embedding_processor: Any,
    embedding_model: Any,
    prompts: List[str],
    display_image: bool = False,
    output_dir: str = "outputs",
) -> List[Tuple[str, float]]:
    """Compute Wasserstein distances between original and generated images.

    Args:
        input_image_path (str): Path to the original input image.
        generated_images (List[Tuple[bool, str]]): Generated image info.
        embedding_processor (Any): Preprocessing object for embeddings.
        embedding_model (Any): Model used for embedding extraction.
        prompts (List[str]): Prompts associated with generated images.
        display_image (bool): Whether to display generated images.
        output_dir (str): Directory to saving computed distances.

    Returns:
        List[Tuple[str, float]]: Image indices and distance values.
    """
    os.makedirs(output_dir, exist_ok=True)

    orig_emb = extract_image_embedding(input_image_path, embedding_processor, embedding_model)

    distances = []
    for idx, ((gen_flag, img_path), text) in enumerate(zip(generated_images, prompts)):

        if gen_flag is not None and not gen_flag:
            dist = float('inf')
        else:
            gen_emb = extract_image_embedding(img_path, embedding_processor, embedding_model)

            if gen_emb is None or orig_emb is None:
                raise ValueError(
                    "One or both embeddings are None. Check embedding extraction."
                )

            gen_emb = np.asarray(gen_emb)
            orig_emb = np.asarray(orig_emb)

            if gen_emb.size == 0 or orig_emb.size == 0:
                raise ValueError(
                    "Embeddings are empty. Cannot compute Wasserstein distance."
                )

            # Compute Wasserstein distance
            dist = wasserstein_distance(gen_emb, orig_emb)

        distances.append((str(idx), dist))
        if display_image:
            gen_img = Image.open(img_path)
            # Display result
            logger.info("Perturbation:")
            logger.info(f"Perturbed Text: {text}")
            logger.info(f"Wasserstein distance (generated vs orig): {dist}")

            plt.figure(figsize=(8, 8))
            plt.imshow(gen_img)
            plt.title(f"Perturbed Text: {text}", fontsize=12)
            plt.axis("off")
            plt.show()

    save_path = os.path.join(output_dir, "WD_dists_generated_vs_orig.npy")
    np.save(save_path, np.array(distances))
    logger.info("All generated embeddings and Wasserstein distances saved.")

    return distances


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
# Train Model
# --------------------------------------------------------------------------- #

def get_model_and_importances(model: Any, method: str) -> np.ndarray:
    """Extracts coefficients or feature importances from a trained model.

    Args:
        model: A trained scikit-learn or XGBoost model instance.
        method (str): The method name (e.g., 'lime', 'randomforest').

    Returns:
        np.ndarray: The feature importances or coefficients.

    Raises:
        AttributeError: If the model lacks both '.coef_' and
                        '.feature_importances_'.
    """
    # Linear models use .coef_
    if hasattr(model, 'coef_'):
        return model.coef_

    # Tree-based models use .feature_importances_
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_

    raise AttributeError(
        f"Model trained with '{method}' does not have a "
        "'.coef_' or '.feature_importances_' attribute."
    )


def fit_surrogate_model(
    perturbations: List[np.ndarray],
    similarities: List[Tuple[str, float]],
    wmd_scores: List[Tuple[str, float]],
    seed: Optional[int] = 1024,
    method: str = "xgboost",
    kernel_width: float = 0.25,
    ridge_alpha: float = 1.0,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Fits a surrogate model (Linear or Non-Linear) to the perturbation data.

    Args:
        perturbations (List[np.ndarray]): List of binary perturbation vectors.
        similarities (List[Tuple[str, float]]): List of (text, similarity)
            pairs.
        wmd_scores (List[Tuple[str, float]]): List of (text, distance) pairs.
        seed (Optional[int]): Random seed for model reproducibility.
        method (str): The method to use ('glm_ols', 'glm_ridge',
            'lime_ols', 'lime_ridge', 'baylime', 'randomforest',
            'gradientboosting' or 'xgboost'). Defaults to 'xgboost'.
        kernel_width (float): Width for the exponential kernel (used for
            weighting).
        ridge_alpha (float): Regularization strength (alpha) for Ridge models.

    Returns:
        Tuple[Any, np.ndarray, np.ndarray]: (trained_model,
            feature_contributions, weights)

    Raises:
        ValueError: If an unknown method is provided.
    """
    # 1. Prepare Data
    X = np.vstack(perturbations)
    y = np.array([s for _, s in similarities])
    dvals = np.array([d for _, d in wmd_scores])
    n_samples = len(y)

    # 2. Determine Weights
    is_global_ols = method == SurrogateMethod.GLM_OLS.value
    is_global_ridge = method == SurrogateMethod.GLM_RIDGE.value
    is_global = is_global_ols or is_global_ridge

    if is_global:
        # Global models (GLM_OLS, GLM_RIDGE) are unweighted
        weights = np.ones(n_samples)
    else:
        # All LIME and Non-Linear models use proximity weighting
        # Weight = sqrt(exp(-distance^2 / sigma^2))
        weights = np.sqrt(np.exp(-(dvals ** 2) / (kernel_width ** 2)))

    logger.debug(f"Weights (head of array): {weights[:5]} (Method: {method})")

    # 3. Model Dispatch and Fitting

    # Define model initialization based on method
    model_map = {
        SurrogateMethod.LIME.value: LinearRegression(),
        SurrogateMethod.LIME_RIDGE.value: Ridge(alpha=ridge_alpha, random_state=seed),
        SurrogateMethod.GLM_OLS.value: LinearRegression(),
        SurrogateMethod.GLM_RIDGE.value: Ridge(alpha=ridge_alpha, random_state=seed),
        SurrogateMethod.BAYLIME.value: BayesianRidge(),
        SurrogateMethod.RANDOMFOREST.value: RandomForestRegressor(random_state=seed),
        SurrogateMethod.GRADIENT_BOOSTING.value: GradientBoostingRegressor(random_state=seed),
        SurrogateMethod.XGBOOST.value: XGBRegressor(random_state=seed, verbosity=0),
    }

    model = model_map.get(method, None)

    if model is None:
        raise ValueError(
            f"Unknown method: {method}. Please use one of the values from"
            " SurrogateMethod."
        )

    # Fit the model: Pass sample_weight only if it's not a global (unweighted) model.
    if is_global:
        # GLM_OLS and GLM_RIDGE are unweighted global fits
        model.fit(X, y)
    else:
        # All LIME and Non-Linear models use sample_weight
        model.fit(X, y, sample_weight=weights)

    # 4. Extract Feature Contributions
    feature_contributions = get_model_and_importances(model, method)

    # 5. Standardized Logging
    is_linear_model = method in [
        SurrogateMethod.GLM_OLS.value, SurrogateMethod.GLM_RIDGE.value, 
        SurrogateMethod.LIME.value, SurrogateMethod.LIME_RIDGE.value, 
        SurrogateMethod.BAYLIME.value
    ]

    # 6. Standardized Logging
    if is_linear_model:
        # Log coefficients with higher precision for linear models
        logger.info(f"Coefficients ({method}): {feature_contributions}")
    else:
        # Log feature importances for non-linear models
        logger.info(f"Feature Importances ({method}): {feature_contributions}")

    return model, feature_contributions, weights


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(
    trained_model: Any,
    feature_contributions: np.ndarray,
    weights: np.ndarray,
    similarities: List[Tuple[str, float]],
    perturbations: List[np.ndarray]
) -> Dict[str, float]:
    """Compute metrics using scikit-learn and NumPy.

    Supports Linear and Non-Linear models.

    Args:
        trained_model: Trained model (Linear or Non-Linear).
        feature_contributions (np.ndarray): feature_contributions.
        weights (np.ndarray): Sample weights used during training.
        similarities (List[Tuple[str, float]]): List of (text, similarity)
            pairs.
        perturbations (List[np.ndarray]): List/Array of perturbation vectors.

    Returns:
        Dict[str, float]: Dictionary of computed metrics.
    """
    # 1. Prepare inputs
    y_true = np.array([s for _, s in similarities])

    # Handle input type for predict (BayesianRidge expects array)
    X = np.vstack(perturbations) if isinstance(perturbations, list) else perturbations
    y_pred = trained_model.predict(X).ravel()

    # Core Metrics via scikit-learn
    # Note: We calculate weighted metrics even for GLM (where weights=1)
    # to maintain interface consistency.
    mse = mean_squared_error(y_true, y_pred, sample_weight=weights)
    mae = mean_absolute_error(y_true, y_pred, sample_weight=weights)
    r2 = r2_score(y_true, y_pred, sample_weight=weights)

    # derived Metrics (NumPy)
    n = len(y_true)
    p = len(feature_contributions)
    diff = y_true - y_pred

    mean_loss = np.abs(np.mean(y_true) - np.mean(y_pred))
    mean_l1 = np.mean(np.abs(diff))
    mean_l2 = np.mean(diff ** 2)

    weighted_l1_norm_n = np.sum(weights * np.abs(diff)) / n
    weighted_l2_norm_n = np.sum(weights * (diff ** 2)) / n

    # Weighted Adjusted R-squared
    if n > p + 1:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        adj_r2 = np.nan

    return {
        "Weighted Mean Squared Error (MSEω)": mse,
        "Weighted Mean Absolute Error (MAEω)": mae,
        "Weighted R-squared (R²ω)": r2,
        "Weighted Adjusted R-squared (R²ω_adj)": adj_r2,
        "Mean Loss (Lm)": mean_loss,
        "Mean L1 Loss (Unweighted MAE)": mean_l1,
        "Mean L2 Loss (Unweighted MSE)": mean_l2,
        "Weighted L1 Loss (Norm by N)": weighted_l1_norm_n,
        "Weighted L2 Loss (Norm by N)": weighted_l2_norm_n,
    }


def find_best_surrogate_model(
    perturbations: List[np.ndarray],
    similarities: List[Tuple[str, float]],
    wmd_scores: List[Tuple[str, float]],
    seed: int = 1024,
    kernel_width: float = 0.25,
    ridge_alpha: float = 1.0,
) -> Tuple[str, float]:
    """
    Finds the surrogate model method (from SurrogateMethod enum) that yields
    the highest Weighted R-squared (R²ω) score.

    Args:
        perturbations (List[np.ndarray]): List of binary perturbation vectors.
        similarities (List[Tuple[str, float]]): List of (text, similarity) pairs.
        wmd_scores (List[Tuple[str, float]]): List of (text, distance) pairs.
        seed (int): Random seed for model reproducibility.
        kernel_width (float): Width for the exponential kernel (used for weighting).
        ridge_alpha (float): Regularization strength (alpha) for Ridge models.

    Returns:
        Tuple[str, float]: A tuple containing the name of the best surrogate method
                           and its corresponding R²ω score.
    """
    best_r2_score = -float('inf')
    best_method_name = ""

    logger.info("Starting search for the best surrogate model...")

    for method_enum in SurrogateMethod:
        method_name = method_enum.value
        logger.info(f"  Testing surrogate method: {method_name}")

        try:
            trained_model, feature_contributions, weights = fit_surrogate_model(
                perturbations=perturbations,
                similarities=similarities,
                wmd_scores=wmd_scores,
                seed=seed,
                method=method_name,
                kernel_width=kernel_width,
                ridge_alpha=ridge_alpha,
            )

            metrics = compute_metrics(
                trained_model=trained_model,
                feature_contributions=feature_contributions,
                weights=weights,
                similarities=similarities,
                perturbations=perturbations,
            )

            current_r2 = metrics.get("Weighted R-squared (R²ω)", -float('inf'))
            logger.info(f"    {method_name} R²ω: {current_r2:.4f}")

            if current_r2 > best_r2_score:
                best_r2_score = current_r2
                best_method_name = method_name

        except Exception as e:
            logger.error(f"  Error processing method {method_name}: {e}")
            continue

    logger.info("\nSearch complete.")
    logger.info(f"Best surrogate method found: {best_method_name} with R²ω = {best_r2_score:.4f}")

    return best_method_name, best_r2_score


def print_metrics(metrics: Dict):
    print('-' * 100)
    print("Fidelity:")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print('-' * 100)


def calculate_stability_score(
    prompt_one: str,
    contributions_one: np.ndarray,
    prompt_two: str,
    contributions_two: np.ndarray
) -> Tuple[float, float]:
    """Calculates stability metrics between two prompts using Jaccard indices.

    This function aligns the feature contributions of two similar prompts based
    on their word tokens and computes both the Generalized Jaccard Similarity
    and Jaccard Distance between their contribution vectors.

    Args:
        prompt_one (str): The first text prompt.
        contributions_one (np.ndarray): Feature contribution scores for prompt_one.
        prompt_two (str): The second text prompt.
        contributions_two (np.ndarray): Feature contribution scores for prompt_two.

    Returns:
        Tuple[float, float]: A tuple containing:
            - Jaccard Similarity: A value between 0 and 1 indicating similarity
              (1.0 means identical importance distributions).
            - Jaccard Distance: A value between 0 and 1 indicating dissimilarity
              (0.0 means identical importance distributions).

    Raises:
        ValueError: If the length of prompts and contribution vectors do not match.
    """
    words_one = prompt_one.split()
    words_two = prompt_two.split()

    if len(words_one) != len(contributions_one) or len(words_two) != len(contributions_two):
        raise ValueError("Length of prompts and contribution vectors must match.")

    # Map words to their contributions
    # Using dictionaries to map words to their corresponding importance scores.
    # Note: Duplicate words will take the value of the last occurrence.
    map_one: Dict[str, float] = dict(zip(words_one, contributions_one))
    map_two: Dict[str, float] = dict(zip(words_two, contributions_two))

    # Create the union of all unique words from both prompts
    unique_words = sorted(list(set(words_one) | set(words_two)))

    # Align vectors based on the union of words.
    # We use absolute values to measure the magnitude of importance.
    vec_one = np.array([abs(map_one.get(w, 0.0)) for w in unique_words])
    vec_two = np.array([abs(map_two.get(w, 0.0)) for w in unique_words])

    # Compute Generalized Jaccard Similarity (Ruzicka similarity)
    # Formula: sum(min(a, b)) / sum(max(a, b))
    numerator = np.sum(np.minimum(vec_one, vec_two))
    denominator = np.sum(np.maximum(vec_one, vec_two))

    if denominator == 0:
        # Both vectors are zero, implying they are identical (empty or zero-weight).
        jaccard_similarity = 1.0
    else:
        jaccard_similarity = numerator / denominator

    jaccard_distance = 1.0 - jaccard_similarity

    return jaccard_similarity, jaccard_distance


def calculate_token_auc(
    text: str,
    scores: List[float],
    truth: List[int]
) -> float:
    """Calculates the Area Under the ROC Curve (AUC) for token importance.

    This function splits the input text into tokens and evaluates how well the
    provided contribution scores align with the ground truth binary labels.

    Args:
        text (str): The raw input prompt/text to be split into tokens.
        scores (List[float]): Predicted importance scores for each token.
        truth (List[int]): Ground truth binary labels (1 for relevant, 0 otherwise).

    Returns:
        float: The calculated ROC AUC score. Returns 0.5 if only one class
            is present in the truth labels (as AUC is undefined).

    Raises:
        ValueError: If the number of tokens derived from the text does not
            match the lengths of the scores or truth lists.
    """
    # Split text into tokens based on whitespace
    tokens = text.split()

    # Validation of input lengths
    if not (len(tokens) == len(scores) == len(truth)):
        raise ValueError(
            f"Dimension mismatch: Found {len(tokens)} tokens, "
            f"{len(scores)} scores, and {len(truth)} truth labels."
        )

    # AUC requires at least one positive and one negative sample
    unique_classes = np.unique(truth)
    if len(unique_classes) < 2:
        return 0.5

    y_true = np.array(truth)
    y_scores = np.array(scores)

    return float(roc_auc_score(y_true, y_scores))


# --------------------------------------------------------------------------- #
# Save data
# --------------------------------------------------------------------------- #

def save_perturbation_data_to_csv(
    perturbations: List[np.ndarray],
    similarities: List[Tuple[str, float]],
    wmd_scores: List[Tuple[str, float]],
    output_path: str = "perturbation_data.csv"
) -> str:
    """Consolidates perturbation data, similarities, and WMD scores and saves
    them to a single CSV file.

    Args:
        perturbations (List[np.ndarray]): List of binary perturbation vectors.
        similarities (List[Tuple[str, float]]): List of (text, similarity) pairs
                                                (y_true equivalents).
        wmd_scores (List[Tuple[str, float]]): List of (text, distance) pairs
                                              (for weights calculation).
        output_path (str): Full path (including filename) where the CSV
                           should be saved.

    Returns:
        str: The full path to the saved CSV file.
    """
    if not perturbations or not similarities or not wmd_scores:
        raise ValueError("All input lists (perturbations, similarities,"
                         " wmd_scores) must be non-empty.")

    # 1. Process Similarities (y_true)
    # The similarities list is (perturbed_text, similarity_score)
    perturbed_texts_sim = [t for t, _ in similarities]
    similarity_scores = [s for _, s in similarities]

    # 2. Process WMD Scores
    # The wmd_scores list is (perturbed_text, wmd_distance)
    wmd_distances = [d for _, d in wmd_scores]
    # Note: We assume the order of similarities and wmd_scores matches
    # the order of perturbations, as implied by the pipeline.

    # 3. Process Perturbations (X matrix)
    # Stack the list of arrays into a 2D matrix
    X_perturbations = np.vstack(perturbations)

    # Create column names for the perturbation vector features (x1, x2, ...)
    n_features = X_perturbations.shape[1]
    feature_cols = [f"x_{i+1}" for i in range(n_features)]

    # 4. Create DataFrame
    # Start with perturbation vectors
    df = pd.DataFrame(X_perturbations, columns=feature_cols)

    # Add text identifiers and target variables
    df.insert(0, 'Perturbed Text', perturbed_texts_sim)
    df['Similarity_Score'] = similarity_scores
    df['WMD_Distance'] = wmd_distances

    # 5. Save to CSV
    save_dir = os.path.dirname(output_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Data successfully saved to: {output_path}")
    return output_path


# --------------------------------------------------------------------------- #
# visualization
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


def plot_actual_vs_predicted(
    linear_model: LinearRegression,
    results_dict: Dict[str, Any],
    save_dir: str = "output",
    filename: str = "actual_vs_predicted_plot.png"
) -> str:
    """
    Creates and saves a beautiful Actual vs Predicted scatter plot
    using pre-computed values from your results dictionary.

    Point size = sample weight (larger = more important sample)

    Args:
        linear_model: Trained LinearRegression model.
        results_dict: Your full results dictionary (as returned by your model)
        save_dir: Where to save the plot
        filename: Name of the output image

    Returns:
        str: Full path to the saved plot
    """
    y_true = np.array([s for _, s in results_dict["similarities"]])
    y_pred = linear_model.predict(results_dict["perturbations"]).ravel()

    # Extract data from your existing results
    weights = np.array(results_dict["weights"])
    weighted_r2 = results_dict["metrics"]["Weighted R-squared (R²ω)"]
    weighted_adj_r2 = results_dict["metrics"]["Weighted Adjusted R-squared (R²ω_adj)"]

    # Optional: fallback if y_true/y_pred not stored yet
    if y_true is None or y_pred is None:
        raise ValueError("results_dict must contain 'y_true' and 'y_pred' (or add them before calling)")

    # Normalize weights for point sizes (50 to 500 looks great)
    if weights.max() > 0:
        point_sizes = (weights / weights.max()) * 450 + 50
    else:
        point_sizes = np.full_like(weights, 100)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        y_true, y_pred,
        s=point_sizes,
        c=weights,
        cmap='plasma',
        alpha=0.75,
        edgecolors='black',
        linewidth=0.5
    )

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min()) * 0.98
    max_val = max(y_true.max(), y_pred.max()) * 1.02
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect Prediction')

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.set_xlabel('Actual Values', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted Values', fontsize=13, fontweight='bold')
    ax.set_title(
        'Actual vs Predicted Values\n'
        f'Weighted R² = {weighted_r2:.4f}  •  Adjusted R² = {weighted_adj_r2:.4f}',
        fontsize=15, fontweight='bold', pad=20
    )

    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Sample Weight (Importance)', rotation=270, labelpad=20, fontsize=11)

    # Save
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close(fig)

    print(f"Plot saved: {save_path}")
    return save_path


def plot_bar_chart(
    data: dict,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    width: int = 800,
    height: int = 600,
    image_save_path: str | None = None
) -> str | None:
    """
    Generates and optionally saves a Plotly bar chart for single values per category.

    Args:
        data (dict): A dictionary where keys are keywords (strings)
                     and values are single numerical data points.
        title (str): The title of the bar chart.
        xaxis_title (str): The label for the x-axis.
        yaxis_title (str): The label for the y-axis.
        width (int): The width of the plot in pixels.
        height (int): The height of the plot in pixels.
        image_save_path (str | None): The file path to save the plot as a static image.

    Returns:
        str | None: The file path to the saved plot, or None if not saved.
    """
    words = list(data.keys())
    # Assuming values are single-element lists, extract the single value
    contributions = [values[0] for values in data.values()]

    fig = go.Figure(data=[go.Bar(x=words, y=contributions, marker_color='skyblue')])

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=False,
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=14, family="Arial", color="black", weight="bold")
        ),
        yaxis=dict(
            tickfont=dict(size=12, family="Arial", color="black", weight="bold")
        ),
        title_font=dict(size=16, family="Arial", color="black", weight="bold"),
        width=width,
        height=height
    )

    if image_save_path:
        save_dir = os.path.dirname(image_save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        pio.write_image(fig, image_save_path)
        print(f"Bar chart saved to: {image_save_path}")

        bar_chart_img = plt.imread(image_save_path)

        plt.figure(figsize=(width // 100, height // 100))
        plt.imshow(bar_chart_img)
        plt.axis("off")
        plt.show()

        return image_save_path

    return None


def plot_box_plot(
    data_updated: dict,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    width: int = 800,
    height: int = 800,
    image_save_path: str | None = None
) -> str | None:
    """
    Generates and optionally saves a Plotly box plot.

    Args:
        data_updated (dict): A dictionary where keys are keywords (strings)
                             and values are lists of numerical data points.
        title (str): The title of the box plot.
        xaxis_title (str): The label for the x-axis.
        yaxis_title (str): The label for the y-axis.
        width (int): The width of the plot in pixels.
        height (int): The height of the plot in pixels.
        image_save_path (str | None): The file path to save the plot as a static image.

    Returns:
        str | None: The file path to the saved plot, or None if not saved.
    """
    fig = go.Figure()

    for keyword, values in data_updated.items():
        fig.add_trace(go.Box(y=values, name=keyword, boxpoints="all"))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=False,
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=14, family="Arial", color="black", weight="bold")
        ),
        yaxis=dict(
            tickfont=dict(size=12, family="Arial", color="black", weight="bold")
        ),
        title_font=dict(size=16, family="Arial", color="black", weight="bold"),
        width=width,
        height=height
    )

    if image_save_path:
        save_dir = os.path.dirname(image_save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        pio.write_image(fig, image_save_path)
        print(f"Box plot saved to: {image_save_path}")

        box_plot_img = plt.imread(image_save_path)

        plt.figure(figsize=(width // 100, height // 100))
        plt.imshow(box_plot_img)
        plt.axis("off")
        plt.show()

        return image_save_path

    return None


def plot_stability_visualization(
    prompt_one: str,
    contributions_one: np.ndarray,
    prompt_two: str,
    contributions_two: np.ndarray,
    width: float = 12.0,
    height: float = 8.0,
    save_path: str | None = None
) -> None:
    """Visualizes the stability flow between two prompts and a generative model.

    Plots Prompt 1 (top) and Prompt 2 (bottom) as heatmaps, with a 'Generative
    Model' node in the center. Edges are drawn from Prompt 1 words to the model,
    and from the model to Prompt 2 words.

    Args:
        prompt_one (str): The first text prompt.
        contributions_one (np.ndarray): Feature contributions for prompt_one.
        prompt_two (str): The second text prompt.
        contributions_two (np.ndarray): Feature contributions for prompt_two.
        width (float): Figure width in inches.
        height (float): Figure height in inches.
        save_path (str | None): Optional path to save the resulting plot.

    Returns:
        None
    """
    words_one = prompt_one.split()
    words_two = prompt_two.split()

    # Normalize scores for coloring (global normalization across both prompts)
    all_scores = np.concatenate([contributions_one, contributions_two])
    denom = np.max(np.abs(all_scores))
    if denom == 0:
        denom = 1e-8

    # Helper to get color
    cmap = plt.cm.ScalarMappable(cmap=plt.cm.bwr)
    cmap.set_clim(0, 1)

    def get_color(score):
        norm_score = 0.5 * score / denom + 0.5
        r, g, b, _ = cmap.to_rgba(norm_score, bytes=True)
        return f"#{r:02x}{g:02x}{b:02x}"

    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Layout Configuration
    y_prompt_one = 85
    y_model = 50
    y_prompt_two = 15

    # Calculate horizontal spacing
    # We distribute words evenly across the width (10 to 90)
    def calculate_x_positions(num_words):
        return np.linspace(10, 90, num_words)

    x_pos_one = calculate_x_positions(len(words_one))
    x_pos_two = calculate_x_positions(len(words_two))
    x_model = 50  # Center

    # Draw Generative Model (Center)
    model_box = FancyBboxPatch(
        (x_model - 5, y_model - 3), 10, 6,
        boxstyle="round,pad=0.2",
        fc="#E0E0E0", ec="black", lw=2
    )
    ax.add_patch(model_box)
    ax.text(x_model, y_model, "Generative\nModel", 
            ha="center", va="center", fontsize=12, fontweight="bold")

    # Draw Prompt One (Top)
    for i, (word, score, x) in enumerate(zip(words_one, contributions_one, x_pos_one)):
        color = get_color(score)

        # Word Box
        ax.text(
            x, y_prompt_one, word,
            bbox=dict(facecolor=color, pad=5.0, linewidth=1, boxstyle="round,pad=0.5"),
            fontsize=12, ha="center"
        )

        # Score Value
        ax.text(
            x, y_prompt_one - 5, f"{score:.2f}",
            fontsize=9, ha="center"
        )

        # Edge: Prompt One -> Model
        # Starting slightly below the score
        ax.annotate(
            "",
            xy=(x_model, y_model + 3),  # Target (Top of model box)
            xytext=(x, y_prompt_one - 6),   # Source (Bottom of score)
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5, shrinkA=5, shrinkB=5)
        )

    # Draw Prompt Two (Bottom)
    for i, (word, score, x) in enumerate(zip(words_two, contributions_two, x_pos_two)):
        color = get_color(score)

        # Word Box
        ax.text(
            x, y_prompt_two, word,
            bbox=dict(facecolor=color, pad=5.0, linewidth=1, boxstyle="round,pad=0.5"),
            fontsize=12, ha="center"
        )

        # Score Value
        ax.text(
            x, y_prompt_two - 5, f"{score:.2f}",
            fontsize=9, ha="center"
        )

        # Edge: Model -> Prompt Two
        # Target is top of the word box (approx y_prompt_two + padding)
        ax.annotate(
            "",
            xy=(x, y_prompt_two + 2),    # Target (Top of word box)
            xytext=(x_model, y_model - 3),   # Source (Bottom of model box)
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5, shrinkA=5, shrinkB=5)
        )

    # Titles
    ax.text(50, 95, "Prompt 1 Source", fontsize=14, fontweight="bold", ha="center")
    ax.text(50, 5, "Prompt 2 Target", fontsize=14, fontweight="bold", ha="center")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_importance_roc_curve(
    truth: List[int],
    scores: List[float],
    title: str = "ROC Curve - Token Importance",
    save_path: Optional[str] = None
) -> None:
    """Plots the Receiver Operating Characteristic (ROC) curve for importance scores.

    This visualization helps evaluate how well the model's contribution scores
    distinguish between ground-truth relevant and irrelevant tokens.

    Args:
        truth (List[int]): Ground truth binary labels (1 for relevant, 0 otherwise).
        scores (List[float]): Predicted importance/contribution scores.
        title (str): Title of the plot.
        save_path (Optional[str]): If provided, saves the plot to this file path.

    Returns:
        None
    """
    y_true = np.array(truth)
    y_scores = np.array(scores)

    # Check if both classes are present
    if len(np.unique(y_true)) < 2:
        print("Warning: ROC curve cannot be plotted with only one class in truth labels.")
        return

    # Calculate FPR, TPR and Area under the curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))

    # Plot the ROC curve
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.4f})")

    # Plot the diagonal baseline (random classifier)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (Irrelevant tokens marked as important)")
    plt.ylabel("True Positive Rate (Relevant tokens correctly identified)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


# --------------------------------------------------------------------------- #
# High-level pipeline
# --------------------------------------------------------------------------- #

def run_interpretability_pipeline(
    prompt: str,
    input_image_path: str,
    model_name: str,
    model_txt: Optional[Any] = None,
    output_dir: str = "output",
    num_perturb: int = 10,
    low_threshold: int = 70,
    high_threshold: int = 200,
    seed: int = 1024,
    kernel_width: float = 0.25,
    ridge_alpha: float = 1.0,
    plot_visualization: bool = True,
    mode: str = "linear",
    surrogate_method: str = "xgboost",
    api_key: Optional[str] = None,
    batch: bool = False,
    find_best_method: bool = False,
) -> Dict[str, Any]:
    """
    Run the full image-text interpretability pipeline.

    This pipeline performs text perturbation, image generation, embedding
    comparison, similarity normalization, and surrogate model fitting to
    explain prompt-level influence on image generation.

    Args:
        prompt (str): Input prompt describing the desired image edit.
        input_image_path (str): Path to the original input image.
        model_name (str): Name of the image generation model.
        model_txt (Optional[Any]): Word embedding model (e.g., Word2Vec).
        output_dir (str): Directory for saving all outputs.
        num_perturb (int): Number of text perturbations to generate.
        low_threshold (int): Canny edge detector low threshold.
        high_threshold (int): Canny edge detector high threshold.
        seed (int): Global random seed for reproducibility.
        kernel_width (float): Kernel width for similarity weighting.
        ridge_alpha (float): Regularization strength for Ridge models.
        plot_visualization (bool): Whether to generate visual outputs.
        mode (str): Similarity normalization mode ("linear" or "inverse").
        surrogate_method (str): Surrogate model type.
        api_key (Optional[str]): API key for remote image models.
        batch (bool): Whether to use batch image generation (Gemini only).
        find_best_method (bool): If True, dynamically finds the best surrogate
            model method based on R²ω score.

    Returns:
        Dict[str, Any]: Dictionary containing perturbations, similarities,
        surrogate model outputs, metrics, and file paths.
    """
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
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True)
    # Move model to CUDA if available, otherwise keep on CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

    if model_txt is None:
        logger.info("Loading Google news Word2Vec model...")
        # TODO: Find Better cache solution for this model
        model_txt = load_google_news_vectors()
        # Precompute and cache vector norms to speed up similarity/WMD calculations
        model_txt.fill_norms(force=True)

    logger.info("Preparing output directory...")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_image_path):
        raise FileNotFoundError(
            f"Input image not found at {input_image_path}"
        )

    logger.info("Validating input prompt...")
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string.")

    normalized_prompt = prompt.strip()

    if not normalized_prompt:
        raise ValueError("Prompt cannot be empty or whitespace only.")

    if len(normalized_prompt) < 10 and normalized_prompt.split() >= 3:
        raise ValueError(
            "Prompt is too short for reliable image editing.\n"
            f"Provided prompt ({len(normalized_prompt)} chars): "
            f"\"{prompt}\"\n"
            "Please use a more descriptive prompt "
            "(at least 18-20 characters)."
        )

    logger.info("Generating text perturbations...")
    responses, perturbations = generate_perturbations(
        text=normalized_prompt,
        num_perturb=num_perturb,
    )

    total_cost = 0.0
    huggingface_pipe = None
    mask_model = None
    mask_model_weights = None

    # Initialize HuggingFace pipe if model_name is instruct-pix2pix
    if "instruct-pix2pix" in model_name.lower():
        model_name = "timbrooks/instruct-pix2pix"
        logger.info("Initializing StableDiffusionInstructPix2PixPipeline...")
        huggingface_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_name, dtype=torch.float16, safety_checker=None
        )
        huggingface_pipe.to(device)
        huggingface_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            huggingface_pipe.scheduler.config
        )
    elif "stable-diffusion-2-inpainting" in model_name.lower():
        model_name = "sd2-community/stable-diffusion-2-inpainting"
        logger.info("Initializing StableDiffusionInpaintPipeline...")
        huggingface_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_name, dtype=torch.float16
        )
        huggingface_pipe.to(device)
        huggingface_pipe.enable_attention_slicing()

        logger.info("Loading DeepLabV3+ segmentation model...")
        mask_model_weights = DeepLabV3_ResNet101_Weights.DEFAULT
        mask_model = deeplabv3_resnet101(weights=mask_model_weights)
        mask_model.to(device)

    logger.info("Starting image generation step...")
    if batch and "gemini" in model_name.lower():
        if api_key is None:
            raise ValueError("API key is required for Gemini models.")
        logger.info(
            "Using batch image generation via Gemini API."
        )
        batch_job_name = submit_gemini_batch_job(
            api_key=api_key,
            image_path=input_image_path,
            text_list=responses,
            model_name=model_name,
            seed=seed,
        )

        generated_images_paths, total_cost = retrieve_gemini_batch_results(
            api_key=api_key,
            job_name=batch_job_name,
            text_list=responses,
            model_name=model_name,
            output_dir=output_dir
        )
    else:
        generated_images_paths, total_cost = generate_images_from_prompts(
            input_image_path=input_image_path,
            prompts=responses,
            model_name=model_name,
            output_dir=output_dir,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            seed=seed,
            api_key=api_key,
            huggingface_pipe=huggingface_pipe,
            mask_model=mask_model,
            mask_model_weights=mask_model_weights,
        )

    logger.info("Computing Wasserstein distances between images...")
    image_distances = compute_wasserstein_distances(
        input_image_path=input_image_path,
        generated_images=generated_images_paths,
        embedding_model=model,
        embedding_processor=processor,
        prompts=responses,
        output_dir=output_dir,
    )

    logger.info("Computing WMD scores between prompt & perturbations...")
    wmd_scores = compute_wmd_scores(model_txt, normalized_prompt, responses)

    # TODO: In the main code we find similarty with inversed mode
    logger.info("Normalizing similarity scores...")
    sims = normalize_similarities(
        distances=image_distances,
        mode=mode,
    )

    # Dynamically find the best surrogate method if flag is True
    if find_best_method:
        logger.info("Finding the best surrogate model...")
        best_method, best_r2 = find_best_surrogate_model(
            perturbations=perturbations,
            similarities=sims,
            wmd_scores=wmd_scores,
            seed=seed,
            kernel_width=kernel_width,
            ridge_alpha=ridge_alpha,
        )
        surrogate_method = best_method
        logger.info(f"Using best surrogate method: {surrogate_method} (R²ω: {best_r2:.4f})")

    logger.info(f"Training surrogate model using method: {surrogate_method}...")
    trained_model, feature_contributions, weights = fit_surrogate_model(
        perturbations=perturbations,
        similarities=sims,
        wmd_scores=wmd_scores,
        seed=seed,
        method=surrogate_method,
        kernel_width=kernel_width,
        ridge_alpha=ridge_alpha,
    )

    logger.info("Computing metrics...")
    metrics = compute_metrics(
        trained_model=trained_model,
        feature_contributions=feature_contributions,
        weights=weights,
        similarities=sims,
        perturbations=perturbations,
    )

    logger.info("Save variables data to pickle file...")
    save_to_pickle(
        output_path=os.path.join(output_dir, f"{model_name.replace('/', '_')}.pkl"),
        responses=responses,
        perturbations=perturbations,
        image_distances=image_distances,
        wmd_scores=wmd_scores,
        sims=sims,
        mode=mode,
        normalized_prompt=normalized_prompt,
        num_perturb=num_perturb,
        seed=seed,
    )

    logger.info("Saving perturbation data to CSV...")
    csv_path = save_perturbation_data_to_csv(
        perturbations=perturbations,
        similarities=sims,
        wmd_scores=wmd_scores,
        output_path=os.path.join(output_dir, f"perturbation_data_{surrogate_method}.csv")
    )

    results = {
        "prompt": normalized_prompt,
        "csv_output_path": csv_path,
        "responses": responses,
        "perturbations": perturbations,
        "wmd_scores": wmd_scores,
        "similarities": sims,
        "feature_contributions": feature_contributions,
        "weights": weights,
        "metrics": metrics,
        "total_image_generation_cost": total_cost
    }

    logger.info(f"Total cost of generated these images is {total_cost}")
    print_metrics(metrics)

    if plot_visualization:
        logger.info("Visualizing results...")
        words = normalized_prompt.split()
        heatmap_save_path = os.path.join(
            output_dir,
            f"heatmap_{surrogate_method.upper()}.png"
        )
        plot_text_heatmap(words, feature_contributions,
                          f"Text Heatmap ({surrogate_method.upper()})",
                          save_path=heatmap_save_path,)

        plot_actual_vs_predicted(
            trained_model,
            results,
            save_dir=output_dir,
            filename=f"actual_vs_predicted_{surrogate_method}.png"
        )

        data_for_bar_chart = {word: [contribution] for word, contribution in zip(words, feature_contributions)}

        bar_chart_save_path = os.path.join(
            output_dir,
            f"bar_chart_{surrogate_method.upper()}.png"
        )
        plot_bar_chart(
            data_for_bar_chart,
            title=f"Bar Chart of Word Contributions ({surrogate_method.upper()})",
            xaxis_title="Keywords",
            yaxis_title="Contribution Value",
            image_save_path=bar_chart_save_path
        )

        data_updated = {word: [contribution] for word, contribution in zip(words, feature_contributions)}

        box_plot_save_path = os.path.join(
            output_dir,
            f"box_plot_{surrogate_method.upper()}.png"
        )
        plot_box_plot(
            data_updated,
            title=f"Box Plot of Word Contributions ({surrogate_method.upper()})",
            xaxis_title="Keywords",
            yaxis_title="Contribution Value",
            image_save_path=box_plot_save_path
        )

    logger.info("Pipeline execution completed successfully.")

    return results


AUC_DICT = {
    "Deblurring": {
        "ori_exp": "Remove the blurriness from the image.",
        "truth": [1, 0, 1, 0, 0, 0]  # Remove(1), blurriness(1)
    },
    "HazeRemoval": {
        "ori_exp": "Remove the haze from the image.",
        "truth": [1, 0, 1, 0, 0, 0]  # Remove(1), haze(1)
    },
    "Lowlight": {
        "ori_exp": "Enhance the brightness of an image.",
        "truth": [1, 0, 1, 0, 0, 0]  # Enhance(1), brightness(1)
    },
    "NoiseRemoval": {
        "ori_exp": "Remove the noise from the image.",
        "truth": [1, 0, 1, 0, 0, 0]  # Remove(1), noise(1)
    },
    "RainRemoval": {
        "ori_exp": "Remove the rain from the image.",
        "truth": [1, 0, 1, 0, 0, 0]  # Remove(1), rain(1)
    },
    "ShadowRemoval": {
        "ori_exp": "Remove the shadow from the image.",
        "truth": [1, 0, 1, 0, 0, 0]  # Remove(1), shadow(1)
    },
    "SnowRemoval": {
        "ori_exp": "Remove the snow from the image.",
        "truth": [1, 0, 1, 0, 0, 0]  # Remove(1), snow(1)
    },
    "WatermarkRemoval": {
        "ori_exp": "Remove the watermark from the image.",
        "truth": [1, 0, 1, 0, 0, 0]  # Remove(1), watermark(1)
    },
    "Counting": {
        "ori_exp": "Add a cat to the shoe rack",
        "truth": [1, 1, 1, 0, 0, 1, 1]  # Add(1), a(1), cat(1), shoe(1), rack(1)
    },
    "DirectionPerception": {
        "ori_exp": "Add a cat to the bottom of the skateboard",
        "truth": [1, 1, 1, 0, 0, 1, 0, 0, 1]  # Add(1), a(1), cat(1), bottom(1), skateboard(1)
    },
    "ObjectRemoval": {
        "ori_exp": "Remove the red slippers from the image",
        "truth": [1, 0, 1, 1, 0, 0, 0]  # Remove(1), red(1), slippers(1)
    },
    "Replacement": {
        "ori_exp": "replace motorcycle with dog",
        "truth": [1, 1, 0, 1]  # replace(1), motorcycle(1), dog(1)
    },
    "BGReplacement": {
        "ori_exp": "Change the background of this photo to snow",
        "truth": [1, 0, 1, 0, 0, 1, 0, 1]  # Change(1), background(1), photo(1), snow(1)
    },
    "ColorAlteration": {
        "ori_exp": "Change the color of the bear to brown",
        "truth": [1, 0, 1, 0, 0, 1, 0, 1]  # Change(1), color(1), bear(1), brown(1)
    },
    "StyleAlteration": {
        "ori_exp": "Change the image to oil painting style",
        "truth": [1, 0, 1, 0, 1, 1, 1]  # Change(1), image(1), oil(1), painting(1), style(1)
    },
    "RegionAccuracy": {
        "ori_exp": "Change the color of the bear to brown",
        "truth": [1, 0, 1, 0, 0, 1, 0, 1]  # Same as ColorAlteration
    }
}

MODELS_DIR = [
    "I-Pix2Pix",
    "Diffusers_I",
    "I2I-Turbo",
    "GPT-Image-1-Mini",
    "GPT-Image-1.5",
    "SeeDream-4.5",
    "Nano-Banana-Pro",
]


if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY", None)

    # logger.info("Loading Google news Word2Vec model...")
    # # TODO: Find Better cache solution for this model
    # model_txt = load_google_news_vectors()
    # # Precompute and cache vector norms to speed up similarity/WMD calculations
    # model_txt.fill_norms(force=True)

    # # use it for img2img turbo
    # results = run_interpretability_pipeline(
    #     prompt="transform it to become blur",
    #     input_image_path="./1.png",
    #     output_dir="./output",
    #     model_name="edge_to_image",
    #     model_txt=model_txt,
    #     low_threshold=70,  # Control Canny edge sensitivity (lower = more edges)
    #     high_threshold=200,  # Control Canny edge sensitivity (higher = fewer edges)
    #     mode="inverse",
    #     api_key=api_key,
    # )

    # # use it for gemini
    # results = run_interpretability_pipeline(
    #     prompt="transform it to become blur",
    #     input_image_path="./1.png",
    #     output_dir="./output",
    #     model_name="gemini-2.5-flash-image",
    #     model_txt=model_txt,
    #     num_perturb=10,
    #     mode="inverse",
    #     api_key=api_key,
    # )

    low_level_editing: List[str] = ['Deblurring', 'HazeRemoval', 'Lowlight', 'NoiseRemoval', 'RainRemoval', 'ShadowRemoval', 'SnowRemoval', 'WatermarkRemoval']
    high_level_edition: List[str] = ['Counting', 'DirectionPerception', 'ObjectRemoval', 'Replacement', 'BGReplacement', 'ColorAlteration', 'StyleAlteration', 'RegionAccuracy']

    parent_dir = "AUC"
    for model in MODELS_DIR:
        logger.info(f"Start calculate for {model} - low level edition")

        model_path = os.path.join(parent_dir, model, "low")
        for cat in low_level_editing:
            res_path = os.path.join(model_path, cat, f"results_{cat}.pkl")
            res = load_from_pickle(res_path)

            scores = res['feature_contributions']
            text = AUC_DICT[cat]["ori_exp"]
            truth = AUC_DICT[cat]["truth"]

            auc_val = calculate_token_auc(text, scores, truth)

            logger.info(f"Calculated AUC for {model} - {cat}: {auc_val:.4f}")

            plot_importance_roc_curve(truth, scores, title=f"ROC Curve {model} - {cat} for: {text}")

        logger.info(f"Start calculate for {model} - high level edition")

        model_path = os.path.join(parent_dir, model, "high")
        for cat in high_level_edition:
            res_path = os.path.join(model_path, cat, f"results_{cat}.pkl")
            res = load_from_pickle(res_path)

            scores = res['feature_contributions']
            text = AUC_DICT[cat]["ori_exp"]
            truth = AUC_DICT[cat]["truth"]

            auc_val = calculate_token_auc(text, scores, truth)

            logger.info(f"Calculated AUC for {model} - {cat}: {auc_val:.4f}")

            plot_importance_roc_curve(truth, scores, title=f"ROC Curve {model} - {cat} for: {text}")
