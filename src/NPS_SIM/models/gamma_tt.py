# -*- coding: utf-8 -*-
"""
Gamma GLM Throughput Time Prediction Model

This module provides functionality for training a Gamma GLM
for throughput time prediction and using it for inference.
It uses a log link function.
The 'alpha' parameter controls L2 regularization and corresponds to
the F_throughput_model_penalty from the simulation design.
"""

import numpy as np
from sklearn.linear_model import GammaRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

def train_gamma_regression(
    X: np.ndarray,
    y_minutes: np.ndarray,
    alpha: float = 0.1,  # L2 regularization strength (F_throughput_model_penalty)
    feature_names: Optional[List[str]] = None
) -> Dict:
    """
    Train a Gamma GLM with log link on the given data.
    The model predicts throughput_time_minutes.

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y_minutes: Target vector of shape (n_samples,) representing throughput time in minutes.
                   Should be positive.
        alpha: L2 regularization strength (corresponds to F_throughput_model_penalty).
        feature_names: List of feature names.

    Returns:
        Dictionary containing model parameters and metadata.
    """
    if X.shape[0] == 0 or y_minutes.shape[0] == 0:
        logger.error("Cannot train Gamma GLM with empty data.")
        return {"training_successful": False, "error": "Empty data"}

    if X.shape[0] != y_minutes.shape[0]:
        logger.error(
            f"Feature matrix and target vector have different number of samples: {X.shape[0]} vs {y_minutes.shape[0]}."
        )
        return {"training_successful": False, "error": "Mismatch in number of samples"}

    if np.any(y_minutes <= 0):
        logger.warning(
            "GammaRegressor expects positive target values. Clamping non-positive y_minutes to a small positive value (1e-6)."
        )
        y_minutes_processed = np.maximum(y_minutes, 1e-6)
    else:
        y_minutes_processed = y_minutes

    # GammaRegressor with log link. Alpha is for L2 penalty.
    model = GammaRegressor(
        alpha=alpha,
        fit_intercept=True,
        max_iter=10000, # Increased max_iter for potentially harder convergence
        random_state=42 # Added random_state for reproducibility
    )

    try:
        model.fit(X, y_minutes_processed)

        intercept = model.intercept_
        betas = model.coef_.tolist()

        # Predictions are directly on the original scale (minutes)
        predictions_minutes = model.predict(X)

        # Calculate MAE and MSE on the original scale (minutes)
        mae_burnin = mean_absolute_error(y_minutes_processed, predictions_minutes)
        mse_burnin = mean_squared_error(y_minutes_processed, predictions_minutes)

        n_samples = len(y_minutes_processed)
        n_features = X.shape[1]

        model_info = {
            "model_type": "gamma_glm", # Changed model type
            "link_function": "log",
            "intercept": float(intercept),
            "betas": betas,
            "alpha_l2": float(alpha), # L2 penalty (F_throughput_model_penalty)
            "n_features": n_features,
            "n_training_samples": n_samples,
            "mae_burnin": float(mae_burnin),
            "mse_burnin": float(mse_burnin),
            "training_successful": True,
            "feature_names": feature_names if feature_names else ["feature_" + str(i) for i in range(n_features)]
        }
        logger.info(
            f"Gamma GLM training successful. Alpha (L2): {alpha}, MAE (minutes): {mae_burnin:.4f}, MSE (minutes): {mse_burnin:.4f}, Samples: {n_samples}"
        )

    except Exception as e:
        logger.error(f"Exception during Gamma GLM model training: {e}")
        model_info = {"training_successful": False, "error": str(e), "model_type": "gamma_glm", "alpha_l2": alpha}

    return model_info


def predict_with_gamma(features: np.ndarray, model_info: Dict) -> Optional[float]:
    """
    Predict throughput time in minutes using a trained Gamma GLM.

    Args:
        features: Numpy array of features for a single case.
        model_info: Dictionary containing the trained Gamma GLM parameters.

    Returns:
        Predicted throughput time in minutes, or None if prediction fails.
    """
    try:
        intercept = model_info["intercept"]
        betas = np.array(model_info["betas"])

        if len(features) != len(betas):
            logger.warning(
                f"Feature length ({len(features)}) and beta length ({len(betas)}) mismatch for Gamma GLM prediction."
            )
            return None

        # Linear predictor part (eta = X@beta + intercept)
        log_mean_prediction_minutes = intercept + np.dot(features, betas)

        # Apply inverse link function (exp for log link)
        # Clip to avoid extremely large values from exp
        prediction_minutes = np.exp(np.clip(log_mean_prediction_minutes, -700, 700))

        return float(prediction_minutes)

    except Exception as e:
        logger.error(f"Error during Gamma GLM prediction: {e}")
        return None 