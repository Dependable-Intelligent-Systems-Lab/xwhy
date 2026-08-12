"""I2EBench dataset downloader and loader.

Reference:
    I2EBench Dataset: https://github.com/cocoshe/I2EBench
"""

import json
import os
import zipfile

import gdown

from xwhy.logger import logger


def download_i2ebench_dataset(
    url: str = "https://drive.google.com/uc?id=10X2C6INLqhY_hbgnOcUNvBD03P-cpX78",
    output_filename: str = "i2ebench.zip",
    extract_dir: str = "i2ebench",
) -> str:
    """Download the I2EBench dataset from Google Drive and extract it.

    Args:
        url: Google Drive URL of the dataset zip file.
        output_filename: Local filename to save the downloaded zip file.
        extract_dir: Directory where the zip file contents will be extracted.

    Returns:
        str: The path to the extracted directory.

    """
    logger.info("Downloading dataset from: %s to %s", url, output_filename)
    gdown.download(url, output_filename, quiet=False)

    logger.info("Creating extraction directory: %s", extract_dir)
    os.makedirs(extract_dir, exist_ok=True)

    logger.info("Extracting %s to: %s", output_filename, extract_dir)
    with zipfile.ZipFile(output_filename, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    logger.info("Dataset extracted to: %s", extract_dir)

    return extract_dir


def load_i2ebench_data(
    root_dir: str = "i2ebench",
    categories: list[str] | None = None,
    limits_per_category: list[int] | int = 1,
) -> dict[str, list[tuple[str, str]]]:
    """Parse the I2EBench dataset with limits and file validation.

    Args:
        root_dir: The root directory of the dataset.
        categories: A list of category names to process. Defaults to the
            standard 8 categories if None.
        limits_per_category: Defines how many items to load for each category.
            If an int (N), it sets the limit to N for ALL categories.

    Returns:
        dict[str, list[tuple[str, str]]]: A dictionary where keys are category
            names and values are lists of tuples. Each tuple contains
            (full_image_path, prompt).

    Raises:
        FileNotFoundError: If main directories or JSON files are missing.
        ValueError: If the lengths of limits and categories do not match.

    """
    if categories is None:
        categories = [
            "Deblurring",
            "HazeRemoval",
            "Lowlight",
            "NoiseRemoval",
            "RainRemoval",
            "ShadowRemoval",
            "SnowRemoval",
            "WatermarkRemoval",
        ]

    # 1. Validate root directory structure
    edit_data_path = os.path.join(root_dir, "EditBench", "EditData")
    if not os.path.exists(edit_data_path):
        raise FileNotFoundError(f"The directory '{edit_data_path}' does not exist.")

    # 2. Handle limit generation logic
    if isinstance(limits_per_category, int):
        limits = [limits_per_category] * len(categories)
    else:
        limits = limits_per_category

    # 3. Check length alignment
    if len(limits) != len(categories):
        raise ValueError(
            f"Length mismatch: 'limits_per_category' has {len(limits)} elements, "
            f"but 'categories' has {len(categories)} elements."
        )

    # 4. Check existence of all category directories
    for category in categories:
        category_path = os.path.join(edit_data_path, category)
        if not os.path.exists(category_path):
            raise FileNotFoundError(f"Category directory not found: {category_path}")

    # 5. Process data
    dataset_dict: dict[str, list[tuple[str, str]]] = {}

    for index, category in enumerate(categories):
        limit = limits[index]
        category_path = os.path.join(edit_data_path, category)
        json_file_path = os.path.join(category_path, f"{category}.json")
        image_input_dir = os.path.join(category_path, "input")

        try:
            with open(json_file_path, encoding="utf-8") as file:
                data = json.load(file)
        except FileNotFoundError as err:
            raise FileNotFoundError(f"JSON file missing: {json_file_path}") from err

        items_list: list[tuple[str, str]] = []

        for info in data.values():
            # STOP condition: Check if we successfully collected enough items
            if len(items_list) >= limit:
                break

            image_filename = info.get("image")
            prompt = info.get("ori_exp")

            if image_filename and prompt:
                full_img_path = os.path.join(image_input_dir, image_filename)

                # Validation: Check if the image file actually exists
                if not os.path.exists(full_img_path):
                    continue

                items_list.append((full_img_path, prompt))

        dataset_dict[category] = items_list

    return dataset_dict
