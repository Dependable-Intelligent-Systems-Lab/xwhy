"""Image-specific evaluation metrics for explainability.

This module provides functions to calculate coverage metrics for image
explanations, evaluating how well the selected features align with
ground-truth semantic masks.
"""

import numpy as np


class ImageCoverageMetrics:
    """Metrics for evaluating image explanation coverage."""

    @staticmethod
    def calculate_coverage(
        explanation_image: np.ndarray,
        semantic_mask: np.ndarray,
        class_of_interest: int = 1,
    ) -> float:
        """Compute the percentage of the true label covered by the explanation.

        The coverage score rewards the explanation for selecting pixels within
        the class of interest and penalizes it for selecting pixels belonging
        to other objects.

        Args:
            explanation_image: The explanation image or mask (non-zero where
                features are selected). Can be 2D or 3D.
            semantic_mask: The 2D ground truth semantic mask where pixels are
                labeled with integer object IDs (0=Background).
            class_of_interest: The label ID of the object to evaluate.

        Returns:
            float: The computed coverage score.

        Raises:
            ValueError: If the spatial dimensions of the explanation image and
                semantic mask do not match.

        """
        ImageCoverageMetrics._validate_shapes(explanation_image, semantic_mask)

        target_area_mask = semantic_mask == class_of_interest
        total_target_pixels = np.count_nonzero(target_area_mask)

        if total_target_pixels == 0:
            return 0.0

        if explanation_image.ndim >= 3:
            active_2d = np.any(explanation_image > 0, axis=-1)
        else:
            active_2d = explanation_image > 0

        reward_mask = active_2d & target_area_mask
        rewards = np.count_nonzero(reward_mask)

        other_objects_mask = (semantic_mask != 0) & (semantic_mask != class_of_interest)
        penalty_mask = active_2d & other_objects_mask
        penalties = np.count_nonzero(penalty_mask)

        return float((rewards - penalties) / total_target_pixels)

    @staticmethod
    def calculate_weighted_coverage(
        explanation_image: np.ndarray,
        semantic_mask: np.ndarray,
        class_of_interest: int = 1,
    ) -> float:
        """Compute the weighted coverage of the explanation.

        This metric rewards explanation feature importance over the class of
        interest (+1 weight) and penalizes importance over other detected
        objects or background (-1 weight).

        Args:
            explanation_image: The explanation image where pixel values represent
                importance scores. Can be 2D or 3D.
            semantic_mask: The 2D ground truth semantic mask.
            class_of_interest: The label ID of the object to evaluate.

        Returns:
            float: The computed weighted coverage score.

        Raises:
            ValueError: If the spatial dimensions do not match.

        """
        ImageCoverageMetrics._validate_shapes(explanation_image, semantic_mask)

        weight_map_2d = np.zeros_like(semantic_mask, dtype=np.float32)
        weight_map_2d[semantic_mask == class_of_interest] = 1.0
        weight_map_2d[semantic_mask != class_of_interest] = -1.0

        if explanation_image.ndim >= 3:
            expanded_weight_map = weight_map_2d
            # Add trailing dimensions to match explanation_image (e.g. channels)
            for _ in range(explanation_image.ndim - 2):
                expanded_weight_map = np.expand_dims(expanded_weight_map, axis=-1)
        else:
            expanded_weight_map = weight_map_2d

        weighted_contribution = explanation_image * expanded_weight_map
        return float(np.sum(weighted_contribution) / explanation_image.size)

    @classmethod
    def evaluate_all(
        cls,
        explanation_image: np.ndarray,
        semantic_mask: np.ndarray,
        class_of_interest: int = 1,
    ) -> tuple[float, float]:
        """Calculate both standard and weighted coverage metrics.

        Args:
            explanation_image: The generated explanation image or mask.
            semantic_mask: The 2D ground truth semantic mask.
            class_of_interest: The label ID of the object to evaluate.

        Returns:
            tuple[float, float]: A tuple containing:
                - coverage (float): Standard coverage score.
                - weighted_coverage (float): Weighted coverage score.

        Raises:
            ValueError: If the spatial dimensions do not match.

        """
        cov = cls.calculate_coverage(
            explanation_image=explanation_image,
            semantic_mask=semantic_mask,
            class_of_interest=class_of_interest,
        )
        w_cov = cls.calculate_weighted_coverage(
            explanation_image=explanation_image,
            semantic_mask=semantic_mask,
            class_of_interest=class_of_interest,
        )
        return cov, w_cov

    @staticmethod
    def _validate_shapes(
        explanation_image: np.ndarray,
        semantic_mask: np.ndarray,
    ) -> None:
        """Validate that the spatial dimensions of the arrays match.

        Args:
            explanation_image: The explanation array (2D or 3D).
            semantic_mask: The 2D semantic mask array.

        Raises:
            ValueError: If the first two dimensions (H, W) do not match.

        """
        if explanation_image.shape[:2] != semantic_mask.shape[:2]:
            raise ValueError(
                f"Spatial dimensions mismatch. Explanation image has spatial "
                f"shape {explanation_image.shape[:2]}, but semantic mask has "
                f"shape {semantic_mask.shape[:2]}."
            )
