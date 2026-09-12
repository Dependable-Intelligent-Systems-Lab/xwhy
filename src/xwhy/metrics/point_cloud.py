"""Point cloud evaluation metrics for explainability."""

from typing import Any, Literal

import numpy as np
import torch

from xwhy.explainers import PointCloudExplainer
from xwhy.logger import logger


def generate_spherical_noise_points(
    center: np.ndarray,
    radius: float,
    num_points: int,
    seed: int | None = None,
) -> np.ndarray:
    """Generate random points uniformly inside a 3D sphere.

    Args:
        center: The center of the sphere as an array of shape (3,).
        radius: The radius of the sphere.
        num_points: The number of points to generate.
        seed: An optional random seed for reproducibility.

    Returns:
        An array of generated points of shape (num_points, 3).

    """
    # Isolate legacy random state to bypass NPY002 while guaranteeing the
    # exact same legacy random sequence for backward-compatible evaluation.
    rng = np.random.RandomState(seed)

    points: list[list[float]] = []
    for _ in range(num_points):
        u, v = rng.uniform(0, 1, 2)

        theta = 2 * np.pi * u
        phi = float(np.arccos(2 * v - 1))
        r = radius * float(np.cbrt(rng.uniform(0, 1)))

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)

        points.append([x + center[0], y + center[1], z + center[2]])

    return np.array(points)


def compute_noisy_explanations(
    sample_input: torch.Tensor,
    sample_label: int,
    model: Any,  # noqa: ANN401
    cluster_labels: np.ndarray | None = None,
    num_clusters: int = 32,
    num_perturbations: int = 1000,
    removal_probability: float = 0.5,
    num_iterations: int = 10,
    num_new_points: int = 30,
    sphere_radius: float = 0.07,
    seed: int = 42,
    clustering_mode: Literal["kmeans", "precomputed"] = "kmeans",
    **explainer_kwargs: Any,  # noqa: ANN401
) -> list[np.ndarray]:
    """Generate noisy point cloud samples and compute their explanations.

    Args:
        sample_input: Original point cloud tensor of shape (B, 3, N) or (3, N).
        sample_label: Ground truth label for the instance.
        model: Trained classification model (custom or Hugging Face).
        cluster_labels: Original cluster labels of shape (N,).
        num_clusters: Number of clusters to group the point cloud into.
        num_perturbations: Number of perturbation masks for the explainer.
        removal_probability: Probability of removing a cluster in perturbations.
        num_iterations: Number of noisy samples to generate.
        num_new_points: Number of noise points to add per iteration.
        sphere_radius: Radius for the noise generation sphere.
        seed: Base random seed for reproducibility.
        clustering_mode: "kmeans" or "precomputed".
        **explainer_kwargs: Additional configurations for PointCloudExplainer
            (e.g., surrogate_type, use_best_surrogate).

    Returns:
        A list of arrays containing the important cluster indices from each run.

    Raises:
        ValueError: If the sample input does not have the expected dimensions.

    """
    all_important_clusters: list[np.ndarray] = []

    sample_np = sample_input.detach().cpu().numpy()
    if sample_np.ndim == 3:
        sample_np = sample_np.squeeze(0)

    if sample_np.shape[-1] != 3:
        raise ValueError(
            f"Expected last dimension to be 3 (X, Y, Z), got {sample_np.shape[-1]}"
        )

    min_coords = sample_np.min(axis=0)
    max_coords = sample_np.max(axis=0)

    for i in range(num_iterations):
        current_seed = seed + i
        rng = np.random.RandomState(current_seed)

        center = np.array([rng.uniform(min_coords[d], max_coords[d]) for d in range(3)])

        new_points = generate_spherical_noise_points(
            center=center,
            radius=sphere_radius,
            num_points=num_new_points,
            seed=current_seed,
        )

        combined_points = np.vstack((sample_np, new_points))
        combined_tensor = torch.from_numpy(combined_points).float()

        if combined_tensor.ndim == 2:
            combined_tensor = combined_tensor.unsqueeze(0)

        combined_labels: np.ndarray | None = None
        if clustering_mode == "precomputed":
            if cluster_labels is None:
                raise ValueError("cluster_labels required for precomputed mode")

            new_labels = np.full(num_new_points, fill_value=cluster_labels.max() + 1)
            combined_labels = np.concatenate([cluster_labels, new_labels])

            # Dynamically determine the correct number of clusters including noise
            actual_num_clusters = len(np.unique(combined_labels))
        else:
            actual_num_clusters = num_clusters

        logger.debug("Sample with Noise: %d", i + 1)

        # Initialize the explainer inside the loop to capture updated cluster counts
        explainer = PointCloudExplainer(
            model=model,
            num_clusters=actual_num_clusters,
            num_perturbations=num_perturbations,
            removal_probability=removal_probability,
            seed=seed,
            clustering_mode=clustering_mode,
            **explainer_kwargs,
        )

        result = explainer.explain(
            instance=combined_tensor,
            sample_label=sample_label,
            cluster_labels=combined_labels,
        )
        all_important_clusters.append(result.important_clusters)

    return all_important_clusters


def calculate_jaccard_stability_score(
    important_clusters_list: list[np.ndarray],
) -> tuple[list[float], float]:
    """Compute the Jaccard similarity across a set of perturbations.

    Args:
        important_clusters_list: A list of arrays, each containing the indices
            of the selected important clusters for a specific evaluation run.

    Returns:
        A tuple containing:
            - A list of Jaccard similarity scores for each perturbation compared
              to the baseline (the first item in the list).
            - The mean Jaccard similarity score across all perturbations.

    Raises:
        ValueError: If the list of important clusters is empty.

    """
    if not important_clusters_list:
        raise ValueError("The list of important clusters cannot be empty.")

    if len(important_clusters_list) == 1:
        return [], 1.0

    base_features = set(important_clusters_list[0])
    jaccard_scores: list[float] = []

    for i, features in enumerate(important_clusters_list[1:], start=1):
        current_features = set(features)
        intersection = len(base_features & current_features)
        union = len(base_features | current_features)

        score = float(intersection / union) if union > 0 else 0.0
        jaccard_scores.append(score)
        logger.debug("Jaccard Similarity with sample %d: %.4f", i, score)

    mean_score = float(np.mean(jaccard_scores))
    logger.debug("Mean Jaccard Similarity: %.4f", mean_score)

    return jaccard_scores, mean_score
