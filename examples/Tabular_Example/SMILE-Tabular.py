"""
Explainability Comparison Module: LIME vs SMILE vs SHAP

This module implements:
1. Data loading and preprocessing
2. Black-box model training
3. Manual LIME implementation
4. SMILE (Wasserstein-based LIME)
5. SHAP and official LIME comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier


# ==============================
# GLOBAL CONFIGURATION
# ==============================

np.random.seed(222)

GRAY_CMAP = LinearSegmentedColormap.from_list(
    "gray_map", [(0.3, 0.3, 0.3), (0.8, 0.8, 0.8)], N=2
)


# ==============================
# DATA LOADING & PREPROCESSING
# ==============================

def load_and_preprocess_data(file_path, feature_columns, target_column):
    """
    Load CSV data and standardize features.

    Args:
        file_path (str): Path to CSV file.
        feature_columns (list[str]): Feature column names.
        target_column (str): Target column name.

    Returns:
        tuple: (X, y, dataframe)
    """
    df = pd.read_csv(file_path)

    X = df[feature_columns].values
    y = df[target_column].values

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    return X, y, df


# ==============================
# VISUALIZATION UTILITIES
# ==============================

def set_plot_style():
    """Set consistent plot style for 2D visualization."""
    plt.axis([-2, 2, -2, 2])
    plt.xlabel("x1")
    plt.ylabel("x2")


def plot_dataset(
    X,
    y=None,
    *,
    cmap=GRAY_CMAP,
    point=None,
    point_style=None,
    scatter_kwargs=None,
    show=True,
):
    """
    Plot dataset or single point with flexible matplotlib-style arguments.

    Supports:
    - Full dataset: shape (n_samples, 2)
    - Single point: shape (2,)

    Args:
        X (np.ndarray or list or float): Input data.
        y (np.ndarray or float, optional): Labels or second coordinate.
        cmap (Colormap, optional): Colormap.
        point (np.ndarray, optional): Additional highlighted point.
        point_style (dict, optional): Style for highlighted point.
        scatter_kwargs (dict, optional): Extra kwargs for plt.scatter.
        show (bool, optional): Whether to display plot.

    Example:
        plot_dataset(X_lime, scatter_kwargs={"s": 2, "c": "black"})
        plot_dataset(instance, scatter_kwargs={"c": "blue"})
    """
    set_plot_style()

    scatter_kwargs = scatter_kwargs or {}

    X = np.asarray(X)

    # ==============================
    # CASE 1: Single point (shape: (2,))
    # ==============================
    if X.ndim == 1 and X.shape[0] == 2:
        plt.scatter(
            X[0],
            X[1],
            **scatter_kwargs,
        )

    # ==============================
    # CASE 2: Dataset (shape: (n, 2))
    # ==============================
    elif X.ndim == 2 and X.shape[1] == 2:
        if y is not None:
            plt.scatter(
                X[:, 0],
                X[:, 1],
                c=y,
                cmap=cmap,
                **scatter_kwargs,
            )
        else:
            plt.scatter(
                X[:, 0],
                X[:, 1],
                **scatter_kwargs,
            )

    else:
        raise ValueError(
            "X must be either shape (n_samples, 2) or (2,)"
        )

    # ==============================
    # OPTIONAL EXTRA POINT
    # ==============================
    if point is not None:
        default_style = {"c": "blue", "marker": "o", "s": 70}
        if point_style:
            default_style.update(point_style)

        plt.scatter(point[0], point[1], **default_style)

    if show:
        plt.show()


def plot_feature_contributions(coefficients, feature_names=None):
    """
    Visualize feature contributions using horizontal bar chart.

    Supports any number of features and separates positive and
    negative contributions dynamically.

    Args:
        coefficients (np.ndarray): Model coefficients (1D array).
        feature_names (list[str], optional): Feature names.

    Returns:
        None
    """
    import plotly.express as px

    coefficients = np.asarray(coefficients)
    num_features = len(coefficients)

    # ==============================
    # Feature names
    # ==============================
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(num_features)]

    # ==============================
    # Vectorized split
    # ==============================
    neg = np.minimum(coefficients, 0)
    pos = np.maximum(coefficients, 0)

    df = pd.DataFrame({
        "feature": feature_names,
        "negative": neg,
        "positive": pos,
    })

    # ==============================
    # Plot (both sides)
    # ==============================
    fig = px.bar(
        df,
        x=["negative", "positive"],
        y="feature",
        orientation="h",
        barmode="relative",
        title="Feature Contributions",
    )

    fig.show()


def plot_method_contributions(
    coefficients,
    feature_names=None,
    method_name="SMILE",
):
    """
    Visualize feature contributions for a given explanation method.

    This function creates a horizontal bar plot using Plotly,
    supporting any number of features and methods (e.g., SMILE, LIME, SHAP).

    Args:
        coefficients (np.ndarray): Feature importance values.
        feature_names (list[str], optional): Feature names.
            If None, default names will be generated.
        method_name (str): Name of the explanation method.

    Returns:
        None
    """
    import numpy as np
    import pandas as pd
    import plotly.express as px

    coefficients = np.asarray(coefficients)
    num_features = len(coefficients)

    # ==============================
    # Feature names handling
    # ==============================
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(num_features)]

    if len(feature_names) != num_features:
        raise ValueError(
            "Length of feature_names must match coefficients"
        )

    # ==============================
    # Create DataFrame
    # ==============================
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": coefficients,
    })

    # Optional: sort for better visualization
    df = df.sort_values("importance", key=np.abs, ascending=True)

    # ==============================
    # Plot
    # ==============================
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        color="feature",
        title=f"{method_name} Feature Contributions",
    )

    fig.show()


def plot_explanation_waterfall(
    coefficients,
    feature_names=None,
    title="Model Explanation (Waterfall)",
    orientation="h",
):
    """
    Create a dynamic waterfall plot for explanation methods
    such as SMILE, LIME, or SHAP coefficients.

    Args:
        coefficients (np.ndarray): Feature importance values.
        feature_names (list[str], optional): Names of features.
            If None, indices will be used.
        title (str): Plot title.
        orientation (str): Plot orientation ("h" or "v").

    Returns:
        plotly.graph_objects.Figure: Waterfall figure.
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    coeffs = np.asarray(coefficients).flatten()
    num_features = len(coeffs)

    # ==============================
    # Handle feature names dynamically
    # ==============================
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(num_features)]

    if len(feature_names) != num_features:
        raise ValueError(
            "feature_names length must match coefficients length"
        )

    # ==============================
    # Build dataframe (clean structure)
    # ==============================
    df = pd.DataFrame(
        {
            "coefficient": coeffs,
            "feature": feature_names,
        }
    )

    # ==============================
    # Create waterfall plot
    # ==============================
    fig = go.Figure(
        go.Waterfall(
            name="explanation",
            orientation=orientation,
            x=df["coefficient"],
            y=df["feature"],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )

    # ==============================
    # Layout
    # ==============================
    fig.update_layout(
        title=title,
        showlegend=False,
    )

    fig.show()


# ==============================
# MODEL TRAINING
# ==============================

def train_random_forest(X, y, n_estimators=100):
    """
    Train Random Forest classifier.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Labels.
        n_estimators (int): Number of trees.

    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X, y)
    return model


# ==============================
# DECISION BOUNDARY
# ==============================

def make_meshgrid(x1, x2, step=0.02):
    """
    Create mesh grid for visualization.

    Args:
        x1 (np.ndarray): Feature 1.
        x2 (np.ndarray): Feature 2.
        step (float): Grid resolution.

    Returns:
        np.ndarray: Grid points.
    """
    x1_min, x1_max = x1.min() - 0.1, x1.max() + 0.1
    x2_min, x2_max = x2.min() - 0.1, x2.max() + 0.1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, step),
        np.arange(x2_min, x2_max, step),
    )

    return np.vstack((xx1.ravel(), xx2.ravel())).T


# ==============================
# BASIC LIME (MANUAL)
# ==============================

def generate_lime_samples(num_samples, num_features):
    """
    Generate LIME perturbations.

    Args:
        num_samples (int): Number of samples.
        num_features (int): Feature count.

    Returns:
        np.ndarray: Perturbations.
    """
    return np.random.uniform(-2, 2, size=(num_samples, num_features))


def compute_lime_weights(instance, samples, kernel_width=0.75):
    """
    Compute LIME weights using Euclidean distance.

    Args:
        instance (np.ndarray): Target instance.
        samples (np.ndarray): Perturbations.
        kernel_width (float): Kernel width.

    Returns:
        np.ndarray: Weights.
    """
    distances = np.sum((instance - samples) ** 2, axis=1)
    weights = np.sqrt(np.exp(-(distances ** 2) / (kernel_width ** 2)))
    return weights


def fit_lime_model(samples, labels, weights):
    """
    Fit linear surrogate model.

    Args:
        samples (np.ndarray): Perturbations.
        labels (np.ndarray): Predictions.
        weights (np.ndarray): Sample weights.

    Returns:
        LinearRegression: Trained model.
    """
    model = LinearRegression()
    model.fit(samples, labels, sample_weight=weights)
    return model


# ==============================
# SMILE (WASSERSTEIN LIME)
# ==============================

def wasserstein_distance_1d(x, y):
    """
    Compute 1D Wasserstein distance using empirical CDF method.

    This implementation preserves the exact logic of the original code.

    Args:
        x (np.ndarray): First sample (length n).
        y (np.ndarray): Second sample (length m).

    Returns:
        float: Wasserstein distance.
    """
    nx = len(x)
    ny = len(y)
    n = nx + ny

    xy = np.concatenate([x, y])
    x_weights = np.concatenate([np.repeat(1 / nx, nx), np.repeat(0, ny)])
    y_weights = np.concatenate([np.repeat(0, nx), np.repeat(1 / ny, ny)])

    sort_idx = np.argsort(xy)
    xy_sorted = xy[sort_idx]
    x_sorted = x_weights[sort_idx]
    y_sorted = y_weights[sort_idx]

    result = 0.0
    ecdf_x = 0.0
    ecdf_y = 0.0

    for i in range(0, n - 2):
        ecdf_x += x_sorted[i]
        ecdf_y += y_sorted[i]

        height = abs(ecdf_y - ecdf_x)
        width = xy_sorted[i + 1] - xy_sorted[i]

        result += height * width

    return result


def wasserstein_distance_p_value(x, y, n_bootstrap=1000):
    """
    Compute Wasserstein distance with bootstrap-based p-value.

    Args:
        x (np.ndarray): First sample.
        y (np.ndarray): Second sample.
        n_bootstrap (int): Number of bootstrap iterations.

    Returns:
        tuple: (p_value, wasserstein_distance)
    """
    import random

    wd = wasserstein_distance_1d(x, y)

    na = len(x)
    nb = len(y)
    n = na + nb

    combined = np.concatenate([x, y])

    bigger = 0

    for _ in range(1, n_bootstrap):
        idx_x = random.sample(range(n), na)
        idx_y = random.sample(range(n), nb)

        wd_boot = wasserstein_distance_1d(
            combined[idx_x],
            combined[idx_y],
        )

        if wd_boot > wd:
            bigger += 1

    p_value = bigger / n_bootstrap

    return p_value, wd


def smile_explainer(
    model,
    instance,
    num_perturbations=500,
    kernel_width=0.2,
    num_distribution_samples=100,
    local_noise=0.05,
    perturbation_noise=0.4,
    epsilon=1.0,
    mode="classification",
):
    """
    SMILE (Wasserstein LIME) implementation.

    This function preserves the exact behavior of the original notebook
    while adding support for epsilon scaling and both classification
    and regression modes.

    Args:
        model: Trained black-box model with predict method.
        instance (np.ndarray): Target instance (shape: [n_features]).
        num_perturbations (int): Number of LIME samples.
        kernel_width (float): Kernel width for weighting.
        num_distribution_samples (int): Samples per distribution.
        local_noise (float): Noise for instance neighborhood.
        perturbation_noise (float): Noise for perturbation distributions.
        epsilon (float): Scaling factor for Wasserstein distance (default=1.0).
        mode (str): "classification" or "regression".

    Returns:
        tuple:
            - X_lime (np.ndarray)
            - y_lime (np.ndarray)
            - weights (np.ndarray)
            - y_linear (np.ndarray)
            - coefficients (np.ndarray)
    """

    if np.abs(np.mean(instance)) > 5:
        raise ValueError(
            "Instance appears not normalized. "
            "Ensure you pass standardized data."
        )

    num_features = len(instance)

    # ==============================
    # Step 1: Generate LIME samples
    # ==============================
    X_lime = np.random.normal(
        0,
        1,
        size=(num_perturbations, num_features),
    )

    # ==============================
    # Step 2: Local distribution around instance
    # ==============================
    instance_distribution = np.zeros(
        (num_distribution_samples, num_features)
    )

    for i in range(num_features):
        instance_distribution[:, i] = (
            instance[i]
            + np.random.normal(0, local_noise, num_distribution_samples)
        )

    # ==============================
    # Step 3: Initialize outputs
    # ==============================
    y_lime = np.zeros((num_perturbations, 1))
    wasserstein_values = np.zeros((num_perturbations, 1))
    weights = np.zeros((num_perturbations, 1))

    # ==============================
    # Step 4: Main loop
    # ==============================
    for idx, sample in enumerate(X_lime):
        # Create distribution around sample
        sample_distribution = np.zeros(
            (num_distribution_samples, num_features)
        )

        for j in range(num_features):
            sample_distribution[:, j] = (
                sample[j]
                + np.random.normal(
                    0,
                    perturbation_noise,
                    num_distribution_samples,
                )
            )

        preds = model.predict(sample_distribution)

        # ==============================
        # Target computation
        # ==============================
        if mode == "classification":
            y_lime[idx] = np.bincount(preds.astype(int)).argmax()
        elif mode == "regression":
            y_lime[idx] = np.mean(preds)
        else:
            raise ValueError("mode must be 'classification' or 'regression'")

        # ==============================
        # Compute Wasserstein distance (per feature)
        # ==============================
        wd_total = 0.0
        for j in range(num_features):
            wd = wasserstein_distance_1d(
                instance_distribution[:, j],
                sample_distribution[:, j],
            )
            wd_total += wd

        wasserstein_values[idx] = wd_total

        # ==============================
        # Compute weight
        # ==============================
        weights[idx] = np.sqrt(
            np.exp(-((epsilon * wd_total) ** 2) / (kernel_width ** 2))
        )

    # ==============================
    # Step 5: Flatten
    # ==============================
    weights = weights.flatten()

    # ==============================
    # Step 6: Fit surrogate model
    # ==============================
    surrogate_model = LinearRegression()
    surrogate_model.fit(X_lime, y_lime, sample_weight=weights)

    y_linear = surrogate_model.predict(X_lime)

    if mode == "classification":
        y_linear = (y_linear < 0.5).flatten()
    else:
        y_linear = y_linear.flatten()

    return (
        X_lime,
        y_lime,
        weights,
        y_linear,
        surrogate_model.coef_.flatten(),
    )


# ==============================
# SHAP METHOD
# ==============================

def plot_shap_explanation(
    model,
    instance,
    class_index=1,
    plot_type="bar",
):
    """
    Compute and visualize SHAP explanation for a given instance.

    Supports bar and waterfall plots.

    Args:
        model: Trained model (e.g., RandomForestClassifier).
        instance (np.ndarray): Input instance of shape (n_features,).
        class_index (int, optional): Target class index.
        plot_type (str, optional): Type of SHAP plot.
            Options:
                - "bar" (default)
                - "waterfall"

    Returns:
        shap.Explanation: Computed SHAP values.
    """
    import shap

    explainer = shap.Explainer(model)
    shap_values = explainer(instance.reshape(1, -1))

    explanation = shap_values[0, :, class_index]

    if plot_type == "bar":
        shap.plots.bar(explanation)

    elif plot_type == "waterfall":
        shap.plots.waterfall(explanation)

    else:
        raise ValueError(
            "plot_type must be either 'bar' or 'waterfall'"
        )

    return shap_values


# ==============================
# OFFICIAL LIME IMPLEMENTATION
# ==============================

def official_lime_explanation(
    X_train,
    instance,
    model,
    feature_names=None,
    class_names=None,
    kernel_width=0.75,
    discretize_continuous=True,
    num_features=None,
    top_labels=1,
):
    """
    Generate explanation using official LIME library (tabular).

    This function provides a flexible wrapper around
    LimeTabularExplainer for consistent usage in experiments.

    Args:
        X_train (np.ndarray): Training data used for LIME fitting.
        instance (np.ndarray): Target instance to explain.
        model: Trained classification model with predict_proba method.
        feature_names (list[str], optional): Feature names.
        class_names (list[str], optional): Class labels.
        kernel_width (float): Kernel width for LIME.
        discretize_continuous (bool): Whether to discretize features.
        num_features (int, optional): Number of features to return.
        top_labels (int): Number of top labels to explain.

    Returns:
        lime.explanation.Explanation: LIME explanation object.
    """
    import lime.lime_tabular
    import numpy as np

    X_train = np.asarray(X_train)

    # ==============================
    # Feature names handling
    # ==============================
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X_train.shape[1])]

    # ==============================
    # num_features default
    # ==============================
    if num_features is None:
        num_features = X_train.shape[1]

    # ==============================
    # Build LIME explainer
    # ==============================
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        kernel_width=kernel_width,
        discretize_continuous=discretize_continuous,
    )

    # ==============================
    # Generate explanation
    # ==============================
    explanation = explainer.explain_instance(
        data_row=np.asarray(instance),
        predict_fn=model.predict_proba,
        num_features=num_features,
        top_labels=top_labels,
    )

    return explanation


# ==============================
# Standard Scaler
# ==============================

class StandardScalerWrapper:
    """
    Simple reusable normalization utility for SMILE/LIME/SHAP consistency.
    """

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0) + 1e-8
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ==============================
# COMPARISON
# ==============================

def compare_methods(lime_coef, smile_coef):
    """
    Compare LIME and SMILE coefficients.

    Args:
        lime_coef (np.ndarray): LIME coefficients.
        smile_coef (np.ndarray): SMILE coefficients.

    Returns:
        pd.DataFrame: Comparison table.
    """
    df = pd.DataFrame({
        "feature": [f"x{i}" for i in range(len(lime_coef))],
        "lime": lime_coef,
        "smile": smile_coef,
    })

    return df


# ==============================
# MAIN EXECUTION PIPELINE
# ==============================

def main():
    """
    Run full explainability pipeline.
    """
    # Load data
    X, y, _ = load_and_preprocess_data(
        file_path="./artificial_data.csv",
        feature_columns=["x1", "x2"],
        target_column="y",
    )

    # Train model
    model = train_random_forest(X, y)

    # Select instance
    instance = np.array([0.8, -0.7])

    # LIME
    lime_samples = generate_lime_samples(500, X.shape[1])
    lime_labels = model.predict(lime_samples)
    lime_weights = compute_lime_weights(instance, lime_samples)

    lime_model = fit_lime_model(
        lime_samples, lime_labels, lime_weights
    )

    # SMILE
    _, _, _, _, smile_coef = smile_explainer(
        model, instance
    )

    # Comparison
    comparison_df = compare_methods(
        lime_model.coef_,
        smile_coef,
    )

    print(comparison_df)


if __name__ == "__main__":
    main()
