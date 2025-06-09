# -*- coding: utf-8 -*-
"""
Dynamic Throughput Time Prediction Model

This module provides functionality for training throughput time prediction models
on burn-in data and using them for inference during the main simulation period.

The model uses enhanced feature engineering with scaling and improved temporal features.

Author: Research Assistant
Created: Based on existing throughput.py
Enhanced: Feature scaling and improved temporal encoding
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize # This import might be unused now
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Import LASSO model functions using a relative import
from .lasso_tt import train_lasso_regression, predict_with_lasso
from .gamma_tt import train_gamma_regression, predict_with_gamma # Added Gamma model

def extract_features_from_case(case_dict: Dict) -> List[float]:
    """
    Extract features from a case dictionary for model training/prediction.
    Enhanced with improved temporal feature engineering and standardization preparation.
    
    Args:
        case_dict: Dictionary containing case information with keys like 'q_dt', 'c_topic', and optionally 'qs_*' features
        
    Returns:
        List of feature values.
    """
    date_and_time = case_dict["q_dt"]
    case_topicidx = case_dict["c_topic"]
    
    # Enhanced temporal features with better engineering
    year_val = float(date_and_time.year)
    # Center year around a reasonable baseline to reduce scale
    year_centered = year_val - 2020.0  # Center around 2020
    
    month_val = float(date_and_time.month) # 1-12 (numerical)
    day_val = float(date_and_time.day)     # 1-31 (numerical)
    weekday_val = date_and_time.weekday() # 0-6 (Monday=0, Sunday=6)
    hour_val = float(date_and_time.hour)   # 0-23 (numerical)
    
    # Cyclical encoding for temporal features to capture periodicity
    # Month cyclical features (captures seasonal patterns)
    month_sin = np.sin(2 * np.pi * month_val / 12.0)
    month_cos = np.cos(2 * np.pi * month_val / 12.0)
    
    # Day of month cyclical features
    day_sin = np.sin(2 * np.pi * day_val / 31.0)
    day_cos = np.cos(2 * np.pi * day_val / 31.0)
    
    # Hour cyclical features (captures daily patterns)
    hour_sin = np.sin(2 * np.pi * hour_val / 24.0)
    hour_cos = np.cos(2 * np.pi * hour_val / 24.0)
        
    # One-hot encode weekday (7 features: weekday_0 to weekday_6)
    weekday_features = [0.0] * 7
    if 0 <= weekday_val <= 6:
        weekday_features[weekday_val] = 1.0

    # Case topic one-hot encoding (10 features)
    # Order: d_2, g_1, j_1, q_3, r_2, w_1, w_2, z_2, z_3, z_4
    case_topics_ordered_for_betas = ["d_2", "g_1", "j_1", "q_3", "r_2", "w_1", "w_2", "z_2", "z_3", "z_4"]
    # Original mapping from general case_topics list to the above specific order:
    # The static model definition in throughput.py uses: ["j_1", "z_3", "q_3", "z_2", "r_2", "z_4", "d_2", "w_2", "g_1", "w_1"]
    # We need to map case_topicidx to the correct position in `topic_dummies` that matches `case_topics_ordered_for_betas`
    
    # This is the order of topics as they appear in the original full list from which case_topicidx is derived
    original_case_topics_list = ["j_1", "z_3", "q_3", "z_2", "r_2", "z_4", "d_2", "w_2", "g_1", "w_1"]
    current_casetopic_str = original_case_topics_list[case_topicidx]

    topic_dummies = [0.0] * 10
    try:
        idx_in_ordered_list = case_topics_ordered_for_betas.index(current_casetopic_str)
        topic_dummies[idx_in_ordered_list] = 1.0
    except ValueError:
        # This case should ideally not happen if all topics in original_case_topics_list are in case_topics_ordered_for_betas
        # If it does, it means there's a mismatch in topic definitions that needs addressing.
        logger.warning(f"Case topic '{current_casetopic_str}' (index {case_topicidx}) not found in defined ordered list for dummies. Topic features will be all zero.")

    # Queue state features (13 features) - Extract if available, otherwise use zeros
    # Apply log transformation to highly skewed features for better scaling
    queue_state_features = []
    queue_feature_names = [
        'qs_agents_available', 'qs_agents_busy', 'qs_agent_utilization',
        'qs_queue_length', 'qs_cases_in_process', 'qs_total_active_cases', 'qs_queue_wait_time_current',
        'qs_recent_completion_rate', 'qs_recent_avg_throughput_time', 'qs_recent_arrival_rate', 'qs_workload_intensity',
        'qs_time_since_last_completion', 'qs_cases_arrived_today'
    ]
    
    # Features that benefit from log transformation (typically right-skewed)
    log_transform_features = {
        'qs_queue_wait_time_current', 'qs_recent_avg_throughput_time', 
        'qs_time_since_last_completion', 'qs_cases_arrived_today'
    }
    
    for feature_name in queue_feature_names:
        raw_value = float(case_dict.get(feature_name, 0.0))
        
        if feature_name in log_transform_features:
            # Log transform with offset to handle zeros
            transformed_value = np.log(1.0 + max(0.0, raw_value))
        else:
            transformed_value = raw_value
            
        queue_state_features.append(transformed_value)

    # Combine all features: Enhanced temporal + OHE weekday + OHE topics + Queue state
    # Total: 1 (year_centered) + 6 (cyclical month/day/hour) + 7 (weekday) + 10 (topics) + 13 (queue) = 37 features
    features = ([year_centered] + 
                [month_sin, month_cos, day_sin, day_cos, hour_sin, hour_cos] + 
                weekday_features + 
                topic_dummies +
                queue_state_features)
    
    return features


def extract_training_data_from_cases(case_list: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract training data from a list of completed cases during burn-in period.
    
    Args:
        case_list: List of case dictionaries from the burn-in period
        
    Returns:
        Tuple of (features_matrix, target_vector) where:
        - features_matrix: numpy array of shape (n_samples, n_features)
        - target_vector: numpy array of shape (n_samples,) with actual throughput times in minutes
    """
    features_list = []
    targets_list = []
    
    total_cases = len(case_list)
    burnin_cases = 0
    closed_cases = 0
    valid_cases = 0
    
    for case in case_list:
        is_burnin = case.get("burn_in", False)
        if is_burnin:
            burnin_cases += 1
        
        is_closed = case.get("status") == "closed"
        if is_closed:
            closed_cases += 1
            
        if is_burnin and is_closed:
            valid_cases += 1
            try:
                features = extract_features_from_case(case)
                
                if case["t_end"] and len(case["t_end"]) > 0:
                    actual_throughput_time = np.max(case["t_end"]) - case["q"] # in days
                    actual_throughput_time_minutes = actual_throughput_time * 24 * 60
                    
                    features_list.append(features)
                    targets_list.append(actual_throughput_time_minutes)
                    
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(f"Skipping case {case.get('i', 'unknown')} for training due to missing data: {e}")
                continue
    
    logger.info(f"Training data extraction: Total cases={total_cases}, Burn-in cases={burnin_cases}, Closed cases={closed_cases}, Valid cases={valid_cases}, Final training samples={len(features_list)}")
    
    if len(features_list) == 0:
        logger.warning("No valid training data found in burn-in cases for dynamic model.")
        return np.array([]).reshape(0,0), np.array([]) # Ensure X has 2 dims even if empty

    features_matrix = np.array(features_list)
    target_vector = np.array(targets_list)
    
    logger.info(f"Extracted {len(features_list)} training samples from burn-in period for dynamic model.")
    
    return features_matrix, target_vector


# def exponential_regression_loss(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
#     """
#     Loss function for exponential regression: y = exp(intercept + X @ beta)
    
#     Args:
#         params: Model parameters [intercept, beta_1, beta_2, ..., beta_n]
#         X: Feature matrix
#         y: Target values
        
#     Returns:
#         Mean squared error loss
#     """
#     intercept = params[0]
#     betas = params[1:]
    
#     # Avoid numerical overflow
#     linear_combination = intercept + X @ betas
#     linear_combination = np.clip(linear_combination, -700, 700)  # Prevent overflow
    
#     predictions = np.exp(linear_combination)
    
#     # Handle potential issues with extreme values
#     predictions = np.clip(predictions, 1e-10, 1e10)
#     y_clipped = np.clip(y, 1e-10, 1e10)
    
#     # Mean squared error
#     mse = np.mean((predictions - y_clipped) ** 2)
    
#     return mse


# def train_exponential_regression(X: np.ndarray, y: np.ndarray) -> Dict:
#     """
#     Train an exponential regression model on the given data.
    
#     Args:
#         X: Feature matrix of shape (n_samples, n_features)
#         y: Target vector of shape (n_samples,)
        
#     Returns:
#         Dictionary containing model parameters and metadata
#     """
#     if len(X) == 0 or len(y) == 0:
#         logger.error("Cannot train model with empty data")
#         return None
    
#     n_features = X.shape[1]
    
#     # Convert to log-space for more stable optimization
#     log_y = np.log(np.maximum(y, 1e-6))  # Avoid log(0)
    
#     # Use simple linear regression in log space instead of complex exponential optimization
#     try:
#         # Add intercept column to X
#         X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
#         # Solve normal equations: beta = (X'X)^(-1)X'y
#         XtX = X_with_intercept.T @ X_with_intercept
#         Xty = X_with_intercept.T @ log_y
        
#         # Add small regularization for numerical stability
#         regularization = 1e-6 * np.eye(XtX.shape[0])
#         XtX_reg = XtX + regularization
        
#         params = np.linalg.solve(XtX_reg, Xty)
        
#         intercept = params[0]
#         betas = params[1:].tolist()
        
#         # Calculate model performance metrics on training data
#         linear_combination = intercept + X @ np.array(betas)
#         linear_combination = np.clip(linear_combination, -700, 700)
#         predictions = np.exp(linear_combination)
        
#         # Calculate MAE and MSE on training data
#         mae_burnin = np.mean(np.abs(predictions - y))
#         mse_burnin = np.mean((predictions - y) ** 2)
#         final_loss = mse_burnin  # Use MSE as the loss
#         n_samples = len(y)
        
#         model_info = {
#             "intercept": float(intercept),
#             "betas": betas,
#             "n_features": n_features,
#             "n_training_samples": n_samples,
#             "final_loss": float(final_loss),
#             "mae_burnin": float(mae_burnin),
#             "mse_burnin": float(mse_burnin),
#             "training_successful": True,
#             "feature_names": ["year", "month", "day", "weekday", "hour", 
#                             "d_2", "g_1", "j_1", "q_3", "r_2", "w_1", "w_2", "z_2", "z_3", "z_4"]
#         }
        
#         logger.info(f"Model training successful. Loss: {final_loss:.4f}, MAE: {mae_burnin:.4f}, MSE: {mse_burnin:.4f}, Samples: {n_samples}")
        
#     except np.linalg.LinAlgError as e:
#         logger.error(f"Linear algebra error during model training: {e}")
#         model_info = {"training_successful": False, "error": str(e)}
#     except Exception as e:
#         logger.error(f"Exception during model training: {e}")
#         model_info = {"training_successful": False, "error": str(e)}
    
#     return model_info


def save_model(model_info: Dict, filepath: Path) -> None:
    """
    Save trained model parameters to JSON file.
    
    Args:
        model_info: Dictionary containing model parameters
        filepath: Path where to save the model JSON file
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(model_info, f, indent=4)
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save model to {filepath}: {e}")


def load_model(filepath: Path) -> Optional[Dict]:
    """
    Load trained model parameters from JSON file.
    
    Args:
        filepath: Path to the model JSON file
        
    Returns:
        Dictionary containing model parameters, or None if loading fails
    """
    try:
        if not filepath.exists():
            logger.warning(f"Model file not found: {filepath}")
            return None
        with open(filepath, 'r') as f:
            model_info = json.load(f)
        
        # Basic check for successful training flag
        if not model_info.get("training_successful", False):
            logger.warning(f"Loaded model from {filepath} indicates training was not successful.")
            # Allow loading to proceed but with a warning, as other parts of model_info might be useful for debugging
            # Or, can return None here if strictly only successfully trained models should be loaded.
            # return None 
            
        logger.info(f"Model loaded from {filepath}")
        return model_info
    except Exception as e:
        logger.error(f"Failed to load model from {filepath}: {e}")
        return None


def predict_TT_dynamic(sigma: Dict, model_info: Optional[Dict] = None) -> Dict:
    """
    Predict throughput time dynamically using a trained model or fallback to static.
    Enhanced with feature scaling support.
    The prediction is returned in DAYS.
    """
    if model_info is None or not model_info.get("training_successful", False):
        # Fallback to static model if no dynamic model info or training failed
        # logger.debug("Dynamic model not available or training failed, using static fallback for TT prediction.")
        return predict_TT_static_fallback(sigma)

    try:
        features = np.array(extract_features_from_case(sigma))
        
        # Apply feature scaling if available
        if model_info.get("feature_scaling_applied", False):
            scaler_mean = np.array(model_info["scaler_mean"])
            scaler_scale = np.array(model_info["scaler_scale"])
            
            if len(features) != len(scaler_mean):
                logger.warning(f"Feature dimension mismatch for scaling: expected {len(scaler_mean)}, got {len(features)}")
                return predict_TT_static_fallback(sigma)
                
            # Apply standardization: (X - mean) / scale
            features_scaled = (features - scaler_mean) / scaler_scale
        else:
            # Use original features if no scaling was applied during training
            features_scaled = features
        
        # Determine which prediction function to use based on model_type
        model_type = model_info.get("model_type")
        prediction_minutes: Optional[float] = None

        if model_type == "lasso":
            prediction_minutes = predict_with_lasso(features_scaled, model_info)
        elif model_type == "gamma_glm":
            prediction_minutes = predict_with_gamma(features_scaled, model_info)
        else:
            logger.warning(f"Unknown dynamic model type: {model_type}. Falling back to static model.")
            return predict_TT_static_fallback(sigma)

        if prediction_minutes is not None:
            # Convert prediction from minutes to days
            sigma["est_throughputtime"] = prediction_minutes / (24 * 60)
            sigma["est_throughputtime_source"] = f"dynamic_{model_type}" 
            # logger.debug(f"Dynamic TT prediction ({model_type}): {sigma['est_throughputtime']:.4f} days for case {sigma.get('i', 'N/A')}")
        else:
            # logger.warning(f"Dynamic TT prediction ({model_type}) returned None. Falling back to static for case {sigma.get('i', 'N/A')}.")
            return predict_TT_static_fallback(sigma)
            
    except Exception as e:
        logger.error(f"Error during dynamic TT prediction with {model_type}: {e}. Falling back to static model.")
        return predict_TT_static_fallback(sigma)
        
    return sigma


def predict_TT_static_fallback(sigma: Dict) -> Dict:
    """
    Fallback to static throughput time prediction.
    Updates 'throughput_model_used' field.
    
    Args:
        sigma: Case dictionary containing case information
        
    Returns:
        Updated sigma dictionary with predicted throughput time
    """
    from models.throughput import predict_TT # Original static model
    sigma["throughput_model_used"] = "static_fallback"
    return predict_TT(sigma)


def calculate_main_period_metrics(case_list: List[Dict], model_info: Dict, F_burn_in: int) -> Tuple[float, float, int]:
    """
    Calculate MAE and MSE for the main simulation period using a trained dynamic model.
    Enhanced with feature scaling support.
    Predictions and actuals are in MINUTES.
    """
    main_period_predictions = []
    main_period_actuals = []
    
    model_type = model_info.get("model_type", "unknown")

    for case in case_list:
        if not case.get("burn_in", True) and case.get("status") == "closed": # Main period, closed cases
            try:
                features = np.array(extract_features_from_case(case))
                
                # Apply feature scaling if available
                if model_info.get("feature_scaling_applied", False):
                    scaler_mean = np.array(model_info["scaler_mean"])
                    scaler_scale = np.array(model_info["scaler_scale"])
                    
                    if len(features) != len(scaler_mean):
                        logger.warning(f"Feature dimension mismatch for scaling in metrics: expected {len(scaler_mean)}, got {len(features)}")
                        continue
                        
                    # Apply standardization: (X - mean) / scale
                    features_scaled = (features - scaler_mean) / scaler_scale
                else:
                    # Use original features if no scaling was applied during training
                    features_scaled = features
                
                prediction_minutes: Optional[float] = None
                if model_type == "lasso":
                    prediction_minutes = predict_with_lasso(features_scaled, model_info)
                elif model_type == "gamma_glm":
                    prediction_minutes = predict_with_gamma(features_scaled, model_info)
                else:
                    logger.warning(f"Cannot calculate main period metrics for unknown model type: {model_type}")
                    continue


                if prediction_minutes is not None and case["t_end"] and len(case["t_end"]) > 0:
                    actual_throughput_time_days = np.max(case["t_end"]) - case["q"]
                    actual_throughput_time_minutes = actual_throughput_time_days * 24 * 60
                    
                    main_period_predictions.append(prediction_minutes)
                    main_period_actuals.append(actual_throughput_time_minutes)
            except Exception as e:
                logger.warning(f"Skipping case for main period metrics due to error: {e}")
                continue
    
    if not main_period_predictions or not main_period_actuals:
        logger.info("No valid cases found for calculating main period model metrics.")
        return float('nan'), float('nan'), 0

    mae_main = mean_absolute_error(main_period_actuals, main_period_predictions)
    mse_main = mean_squared_error(main_period_actuals, main_period_predictions)
    n_main = len(main_period_actuals)
    
    logger.info(f"Main period metrics ({model_type}): MAE={mae_main:.4f} min, MSE={mse_main:.4f} min, N={n_main}")
    
    return mae_main, mse_main, n_main


def train_model_on_burn_in(
    case_list: List[Dict], 
    run_dir: Path,
    throughput_model_type: str = "Lasso",
    model_penalty_alpha: float = 0.01  # Reduced from 0.1 for less aggressive regularization
) -> Optional[Dict]:
    """
    Train a throughput time prediction model using data from the burn-in period.
    Enhanced with feature scaling and improved regularization.
    
    Args:
        case_list: List of completed cases from burn-in period
        run_dir: Directory to save model files
        throughput_model_type: Type of model to train ("Lasso", "gamma_glm")
        model_penalty_alpha: Regularization strength (reduced default for better performance)
        
    Returns:
        Dictionary containing trained model information, or None if training fails
    """
    logger.info(f"Starting dynamic throughput model training ({throughput_model_type}) on burn-in data with penalty alpha={model_penalty_alpha}...")
    
    X_train, y_train = extract_training_data_from_cases(case_list)
    
    if X_train.shape[0] == 0:
        logger.warning("No training data extracted. Dynamic model training aborted.")
        return {"training_successful": False, "error": "No training data from burn-in"}

    # Define feature names based on enhanced feature extraction
    # Enhanced temporal (7) + Weekday (7) + Topics (10) + Queue state (13) = 37 features
    feature_names = (['year_centered'] + 
                    ['month_sin', 'month_cos', 'day_sin', 'day_cos', 'hour_sin', 'hour_cos'] +
                    [f'weekday_{i}' for i in range(7)] +
                    ["d_2", "g_1", "j_1", "q_3", "r_2", "w_1", "w_2", "z_2", "z_3", "z_4"] +
                    ['qs_agents_available', 'qs_agents_busy', 'qs_agent_utilization',
                     'qs_queue_length', 'qs_cases_in_process', 'qs_total_active_cases', 'qs_queue_wait_time_current',
                     'qs_recent_completion_rate', 'qs_recent_avg_throughput_time', 'qs_recent_arrival_rate', 'qs_workload_intensity',
                     'qs_time_since_last_completion', 'qs_cases_arrived_today'])

    # Feature scaling implementation
    logger.info("Applying feature scaling to training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    logger.info(f"Feature scaling applied. Mean normalization: {np.mean(np.abs(scaler.mean_)):.4f}, "
                f"Scale normalization: {np.mean(scaler.scale_):.4f}")

    model_info: Optional[Dict] = None
    
    try:
        if throughput_model_type.lower() == "lasso":
            # Train enhanced LASSO model with feature scaling
            model_info = train_lasso_regression(
                X_train_scaled, y_train, 
                alpha=model_penalty_alpha, 
                feature_names=feature_names
            )
            
            # Add scaling parameters to model info
            model_info["scaler_mean"] = scaler.mean_.tolist()
            model_info["scaler_scale"] = scaler.scale_.tolist()
            model_info["feature_scaling_applied"] = True
            
            logger.info(f"LASSO model trained with feature scaling. "
                       f"Alpha: {model_penalty_alpha}, Features: {len(feature_names)}")
            
        elif throughput_model_type.lower() == "gamma_glm":
            # Train Gamma GLM model with feature scaling
            model_info = train_gamma_regression(
                X_train_scaled, y_train, 
                alpha=model_penalty_alpha, 
                feature_names=feature_names
            )
            
            # Add scaling parameters to model info
            model_info["scaler_mean"] = scaler.mean_.tolist()
            model_info["scaler_scale"] = scaler.scale_.tolist()
            model_info["feature_scaling_applied"] = True
            
            logger.info(f"Gamma GLM model trained with feature scaling. "
                       f"Alpha: {model_penalty_alpha}, Features: {len(feature_names)}")
            
        else:
            logger.error(f"Unknown model type: {throughput_model_type}")
            return {"training_successful": False, "error": f"Unknown model type: {throughput_model_type}"}
            
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return {"training_successful": False, "error": str(e)}
    
    if model_info is None or not model_info.get("training_successful", False):
        logger.error("Model training was not successful.")
        return model_info
    
    # Save model to file
    try:
        models_dir = run_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        model_file = models_dir / "dynamic_throughput_model.json"
        save_model(model_info, model_file)
        logger.info(f"Enhanced dynamic model saved to: {model_file}")
        
    except Exception as e:
        logger.warning(f"Failed to save model to file: {e}")
        # Continue anyway, model_info is still valid for this run
    
    return model_info
