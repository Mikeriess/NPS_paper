# -*- coding: utf-8 -*-
"""
Dynamic Throughput Time Prediction Model

This module provides functionality for training throughput time prediction models
on burn-in data and using them for inference during the main simulation period.

The model uses the same feature set as the static throughput model but allows
for parameter estimation based on observed data during the burn-in period.

Author: Research Assistant
Created: Based on existing throughput.py
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize # This import might be unused now
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Import LASSO model functions using a relative import
from .lasso_tt import train_lasso_regression, predict_with_lasso

def extract_features_from_case(case_dict: Dict) -> List[float]:
    """
    Extract features from a case dictionary for model training/prediction.
    Year is included as a numerical feature.
    Month, day (of month), weekday, and hour are one-hot encoded.
    Case topics are also one-hot encoded.
    
    Args:
        case_dict: Dictionary containing case information with keys like 'q_dt', 'c_topic'
        
    Returns:
        List of feature values.
    """
    date_and_time = case_dict["q_dt"]
    case_topicidx = case_dict["c_topic"]
    
    # Temporal features (raw)
    year_val = float(date_and_time.year)
    month_val = date_and_time.month # 1-12
    day_val = date_and_time.day     # 1-31
    weekday_val = date_and_time.weekday() # 0-6 (Monday=0, Sunday=6)
    hour_val = date_and_time.hour   # 0-23
    
    # One-hot encode month (12 features: month_1 to month_12)
    month_features = [0.0] * 12
    if 1 <= month_val <= 12:
        month_features[month_val - 1] = 1.0
        
    # One-hot encode day of month (31 features: day_1 to day_31)
    day_features = [0.0] * 31
    if 1 <= day_val <= 31:
        day_features[day_val - 1] = 1.0
        
    # One-hot encode weekday (7 features: weekday_0 to weekday_6)
    weekday_features = [0.0] * 7
    if 0 <= weekday_val <= 6:
        weekday_features[weekday_val] = 1.0
        
    # One-hot encode hour (24 features: hour_0 to hour_23)
    hour_features = [0.0] * 24
    if 0 <= hour_val <= 23:
        hour_features[hour_val] = 1.0

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

    # Combine all features: Year (numerical) + OHE month + OHE day + OHE weekday + OHE hour + OHE topics
    features = ([year_val] + 
                month_features + 
                day_features + 
                weekday_features + 
                hour_features + 
                topic_dummies)
    
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
    Predict throughput time (TT) using a dynamically trained model.
    If no dynamic model is provided or if it's invalid, fall back to static prediction.
    
    Args:
        sigma: Dictionary containing case information.
        model_info: Dictionary containing the trained dynamic model parameters.
                    Expected to have 'model_type': 'lasso' for LASSO model.
                    
    Returns:
        sigma: The input dictionary updated with 'est_throughputtime' (in days)
               and 'throughput_model_used' ('dynamic_lasso' or 'static_fallback').
    """
    try:
        if model_info and model_info.get("training_successful") and model_info.get("model_type") == "lasso":
            features_list = extract_features_from_case(sigma)
            features_np = np.array(features_list) # Ensure it's a numpy array

            # Check if features_np needs reshaping for single prediction: (1, n_features)
            # The predict_with_lasso function expects a 1D array of features for a single case.
            # If extract_features_from_case returns a list, np.array(features_list) should be 1D.

            predicted_tt_minutes = predict_with_lasso(features_np, model_info)
            
            if predicted_tt_minutes is not None and predicted_tt_minutes > 0:
                predicted_tt_days = predicted_tt_minutes / (24 * 60)
                sigma["est_throughputtime"] = predicted_tt_days
                sigma["throughput_model_used"] = "dynamic_lasso"
                # logger.debug(f"Dynamic LASSO prediction for case {sigma.get('i', 'unknown')}: {predicted_tt_days:.4f} days")
                return sigma # Return early on successful dynamic prediction
            else:
                logger.warning(f"Dynamic LASSO prediction for case {sigma.get('i', 'unknown')} "
                               f"resulted in {predicted_tt_minutes}. Falling back to static.")
                return predict_TT_static_fallback(sigma)
        
        else: # Fallback conditions
            if model_info: # A model was loaded but not suitable
                if not model_info.get("training_successful"):
                    logger.debug(f"Dynamic model training was not successful. Falling back to static for case {sigma.get('i', 'unknown')}.")
                elif model_info.get("model_type") != "lasso":
                    logger.warning(f"Loaded dynamic model is not of type LASSO (type: {model_info.get('model_type')}). "
                                   f"Falling back to static for case {sigma.get('i', 'unknown')}.")
                # else: # Other model_info issues
                    # logger.debug(f"Problem with loaded dynamic model. Falling back to static for case {sigma.get('i', 'unknown')}.")
            # else: # No model_info provided
                # logger.debug(f"No dynamic model provided. Falling back to static for case {sigma.get('i', 'unknown')}.")
            return predict_TT_static_fallback(sigma)
            
    except Exception as e: # Catch-all for unexpected errors during prediction logic
        logger.error(f"Error in predict_TT_dynamic dispatch: {e}. "
                       f"Using static fallback for case {sigma.get('i', 'unknown')}.")
        return predict_TT_static_fallback(sigma)


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
    Calculate MAE and MSE for the main period cases using the trained dynamic model.
    
    Args:
        case_list: List of all cases from simulation
        model_info: Trained model information dictionary (expected for dynamic model eval)
        F_burn_in: Burn-in period threshold (day number)
        
    Returns:
        Tuple of (mae_main, mse_main, n_main_cases) where:
        - mae_main: Mean Absolute Error on main period cases (in days)
        - mse_main: Mean Squared Error on main period cases (in days^2)
        - n_main_cases: Number of main period cases used for evaluation
    """
    # This function assumes that est_throughputtime (in days) is already populated in cases
    # by predict_TT_dynamic (or static fallback) during the simulation.
    
    if not model_info or not model_info.get("training_successful", False):
        logger.warning("No valid trained dynamic model provided for main period metrics calculation. Metrics will reflect fallback if used.")
        # Depending on requirements, one might still proceed to see performance of whatever model was used (e.g. static)
        # For now, let's assume we are interested in dynamic model's performance when available.
        # If strictly only dynamic, could return nan, nan, 0 if model_info is not the dynamic one.

    predictions_main_days = []
    actuals_main_days = []
    n_main_dynamic_used = 0
    n_main_static_used = 0
    
    for case in case_list:
        # Only use closed cases from main period (after burn-in period ENDS)
        # Assuming F_burn_in is the day number that marks the END of burn-in.
        # So main period cases are those whose queue time 'q' is >= F_burn_in.
        if (not case.get("burn_in", True)) and \
           case.get("status") == "closed" and \
           case.get("q", float('-inf')) >= F_burn_in: # Ensure q exists and is comparable
            
            try:
                predicted_tt_days = case.get("est_throughputtime") # This is already in days
                
                if case.get("t_end") and len(case["t_end"]) > 0 and predicted_tt_days is not None:
                    actual_tt_days = np.max(case["t_end"]) - case["q"] # Already in days
                    
                    predictions_main_days.append(predicted_tt_days)
                    actuals_main_days.append(actual_tt_days)

                    if case.get("throughput_model_used") == "dynamic_lasso":
                        n_main_dynamic_used +=1
                    elif case.get("throughput_model_used") == "static_fallback":
                        n_main_static_used +=1
                else:
                    logger.warning(f"Skipping case {case.get('i', 'unknown')} for main period metrics: missing t_end or predicted_tt.")        
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(f"Skipping case {case.get('i', 'unknown')} in main period metrics due to data issue: {e}")
                continue
    
    if len(predictions_main_days) == 0:
        logger.warning("No valid main period cases found for metrics calculation.")
        return float('nan'), float('nan'), 0
    
    predictions_array_days = np.array(predictions_main_days)
    actuals_array_days = np.array(actuals_main_days)
    
    mae_main_days = np.mean(np.abs(predictions_array_days - actuals_array_days))
    mse_main_days = np.mean((predictions_array_days - actuals_array_days) ** 2)
    n_main_cases = len(predictions_main_days)
    
    logger.info(f"Main period metrics (N={n_main_cases}): MAE={mae_main_days:.4f} days, MSE={mse_main_days:.4f} days^2.")
    logger.info(f"  Dynamic LASSO model used for {n_main_dynamic_used} main period cases.")
    logger.info(f"  Static fallback model used for {n_main_static_used} main period cases.")

    return float(mae_main_days), float(mse_main_days), n_main_cases


def train_model_on_burn_in(case_list: List[Dict], run_dir: Path) -> Optional[Dict]:
    """
    Train a throughput prediction model based on data from the burn-in period.
    This function now trains a LASSO model.
    
    Args:
        case_list: List of all case dictionaries (includes burn-in and main period cases)
        run_dir: Path to the current run's directory for saving the model.
        
    Returns:
        Dictionary containing the trained model's parameters and metadata, or None if training fails.
    """
    logger.info("Starting dynamic throughput model training on burn-in data (using LASSO).")
    
    # extract_training_data_from_cases filters for burn-in cases itself
    X_train, y_train_minutes = extract_training_data_from_cases(case_list) 
    
    if X_train.shape[0] == 0: # Check if any training samples were actually extracted
        logger.warning("No training data extracted from burn-in period (X_train is empty). Cannot train dynamic LASSO model.")
        return None # Indicate failure: no data
        
    # Feature names must match the order in extract_features_from_case
    feature_names = ["year"] + \
                    [f"month_{i}" for i in range(1, 13)] + \
                    [f"day_{i}" for i in range(1, 32)] + \
                    [f"weekday_{i}" for i in range(7)] + \
                    [f"hour_{i}" for i in range(24)] + \
                    ["topic_d_2", "topic_g_1", "topic_j_1", "topic_q_3", "topic_r_2", 
                     "topic_w_1", "topic_w_2", "topic_z_2", "topic_z_3", "topic_z_4"] # Matching the order in case_topics_ordered_for_betas

    if X_train.shape[1] != len(feature_names):
        logger.error(f"Mismatch between number of features in training data ({X_train.shape[1]}) and expected feature_names count ({len(feature_names)}). Features expected: {feature_names}")
        # This indicates a problem in feature extraction or feature_names list.
        return None 

    # Train the LASSO model. Alpha is hardcoded in train_lasso_regression (0.1 by default)
    # but can be overridden if train_lasso_regression is modified or if passed here.
    logger.info(f"Training LASSO model with default alpha on {X_train.shape[0]} samples and {X_train.shape[1]} features.")
    model_info = train_lasso_regression(X_train, y_train_minutes, feature_names=feature_names)
                                         
    if model_info and model_info.get("training_successful"):
        # model_type: "lasso" is already set by train_lasso_regression
        
        # Define where to save the model file
        model_output_dir = run_dir / "models" # Store in a 'models' subdirectory
        model_output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        model_filepath = model_output_dir / "dynamic_throughput_model.json"
        
        save_model(model_info, model_filepath) # save_model also ensures parent dir exists
        logger.info(f"Dynamic LASSO throughput model trained successfully and saved to {model_filepath}")
        logger.info(f"  Model Type: {model_info.get('model_type')}")
        logger.info(f"  LASSO Intercept: {model_info.get('intercept')}")
        # logger.info(f"  LASSO Betas: {model_info.get('betas')}") # Usually too verbose for console
        logger.info(f"  LASSO Alpha: {model_info.get('alpha')}")
        logger.info(f"  Training Samples: {model_info.get('n_training_samples')}")
        logger.info(f"  MAE (burn-in, minutes): {model_info.get('mae_burnin')}")
        logger.info(f"  MSE (burn-in, minutes): {model_info.get('mse_burnin')}")
    else:
        logger.error("Dynamic LASSO throughput model training failed.")
        if model_info and "error" in model_info: # Log error details if available
             logger.error(f"  Error details: {model_info['error']}")
        # Ensure a clear None is returned on failure.
        # The model_info might contain error details, but the function should return None.
        return None 
        
    return model_info
