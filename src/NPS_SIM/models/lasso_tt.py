# -*- coding: utf-8 -*-
"""
LASSO Throughput Time Prediction Model

This module provides functionality for training a LASSO regression model
for throughput time prediction and using it for inference.

The 'alpha' parameter controls L1 regularization strength and corresponds to
the F_throughput_model_penalty from the simulation design.
"""

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

def train_lasso_regression(
    X: np.ndarray, 
    y_minutes: np.ndarray, 
    alpha: float = 0.1, # L1 regularization strength (F_throughput_model_penalty)
    feature_names: Optional[List[str]] = None
) -> Dict:
    """
    Train a LASSO regression model on the given data.
    The model predicts log(throughput_time_minutes).

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y_minutes: Target vector of shape (n_samples,) representing throughput time in minutes.
        alpha: LASSO L1 regularization strength (corresponds to F_throughput_model_penalty).
        feature_names: List of feature names.

    Returns:
        Dictionary containing model parameters and metadata.
    """
    if X.shape[0] == 0 or y_minutes.shape[0] == 0:
        logger.error("Cannot train LASSO model with empty data.")
        return {"training_successful": False, "error": "Empty data"}
    
    if X.shape[0] != y_minutes.shape[0]:
        logger.error(f"Feature matrix and target vector have different number of samples: {X.shape[0]} vs {y_minutes.shape[0]}.")
        return {"training_successful": False, "error": "Mismatch in number of samples"}

    # Log-transform the target variable (throughput time in minutes)
    # Adding a small constant to avoid log(0) or log(negative) if data has such issues.
    log_y_minutes = np.log(np.maximum(y_minutes, 1e-6))

    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000, random_state=42) # Added random_state for reproducibility

    try:
        model.fit(X, log_y_minutes)

        intercept = model.intercept_
        betas = model.coef_.tolist()

        # Make predictions on the log scale
        log_predictions_minutes = model.predict(X)
        # Transform predictions back to the original scale (minutes)
        predictions_minutes = np.exp(log_predictions_minutes)

        # Calculate MAE and MSE on the original scale (minutes)
        # Ensure y_minutes is used here, not log_y_minutes
        mae_burnin = mean_absolute_error(y_minutes, predictions_minutes)
        mse_burnin = mean_squared_error(y_minutes, predictions_minutes)
        
        n_samples = len(y_minutes)
        n_features = X.shape[1]

        model_info = {
            "model_type": "lasso",
            "intercept": float(intercept),
            "betas": betas,
            "alpha_l1": float(alpha), # Clarified alpha is L1 penalty (F_throughput_model_penalty)
            "n_features": n_features,
            "n_training_samples": n_samples,
            "mae_burnin": float(mae_burnin), # MAE on original scale (minutes)
            "mse_burnin": float(mse_burnin), # MSE on original scale (minutes)
            "training_successful": True,
            "feature_names": feature_names if feature_names else ["feature_" + str(i) for i in range(n_features)]
        }
        logger.info(
            f"LASSO model training successful. Alpha (L1): {alpha}, MAE (minutes): {mae_burnin:.4f}, MSE (minutes): {mse_burnin:.4f}, Samples: {n_samples}"
        )

    except Exception as e:
        logger.error(f"Exception during LASSO model training: {e}")
        model_info = {"training_successful": False, "error": str(e), "model_type": "lasso", "alpha_l1": alpha}
    
    return model_info


def predict_with_lasso(features: np.ndarray, model_info: Dict) -> Optional[float]:
    """
    Predict throughput time in minutes using a trained LASSO model.

    Args:
        features: Numpy array of features for a single case.
        model_info: Dictionary containing the trained LASSO model parameters.

    Returns:
        Predicted throughput time in minutes, or None if prediction fails.
    """
    try:
        intercept = model_info["intercept"]
        betas = np.array(model_info["betas"])

        if len(features) != len(betas):
            logger.warning(
                f"Feature length ({len(features)}) and beta length ({len(betas)}) mismatch for LASSO prediction."
            )
            return None

        # Prediction is on the log scale
        log_prediction_minutes = intercept + np.dot(features, betas)
        
        # Transform back to original scale (minutes)
        # Clip to avoid extremely large values from exp, if necessary, though usually handled by model quality
        prediction_minutes = np.exp(np.clip(log_prediction_minutes, -700, 700)) 

        return float(prediction_minutes)

    except Exception as e:
        logger.error(f"Error during LASSO prediction: {e}")
        return None 