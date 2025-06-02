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
from scipy.optimize import minimize
import logging

# Set up logging
logger = logging.getLogger(__name__)


def extract_features_from_case(case_dict: Dict) -> List[float]:
    """
    Extract features from a case dictionary for model training/prediction.
    
    Args:
        case_dict: Dictionary containing case information with keys like 'q_dt', 'c_topic'
        
    Returns:
        List of feature values in the same order used by the static model
    """
    date_and_time = case_dict["q_dt"]
    case_topicidx = case_dict["c_topic"]
    
    # Temporal features
    year = date_and_time.year
    month = date_and_time.month
    day = date_and_time.day
    weekday = date_and_time.weekday()
    hour = date_and_time.hour
    
    # Case topic one-hot encoding (same order as static model)
    case_topics = ["j_1", "z_3", "q_3", "z_2", "r_2", "z_4", "d_2", "w_2", "g_1", "w_1"]
    casetopic = case_topics[case_topicidx]
    
    # Initialize topic dummy variables
    topic_dummies = [0] * 10  # d_2, g_1, j_1, q_3, r_2, w_1, w_2, z_2, z_3, z_4
    
    # Map case topic to dummy variables correctly (fixing the static model bug)
    # The static model incorrectly treats d_2 topic as j_1, but we implement it correctly here
    if casetopic == "d_2":
        topic_dummies[0] = 1  # d_2 = 1 (correct implementation)
    elif casetopic == "g_1":
        topic_dummies[1] = 1  # g_1 = 1
    elif casetopic == "j_1":
        topic_dummies[2] = 1  # j_1 = 1
    elif casetopic == "q_3":
        topic_dummies[3] = 1  # q_3 = 1
    elif casetopic == "r_2":
        topic_dummies[4] = 1  # r_2 = 1
    elif casetopic == "w_1":
        topic_dummies[5] = 1  # w_1 = 1
    elif casetopic == "w_2":
        topic_dummies[6] = 1  # w_2 = 1
    elif casetopic == "z_2":
        topic_dummies[7] = 1  # z_2 = 1
    elif casetopic == "z_3":
        topic_dummies[8] = 1  # z_3 = 1
    elif casetopic == "z_4":
        topic_dummies[9] = 1  # z_4 = 1
    
    # Return features in same order as static model: [year, month, day, weekday, hour, d_2, g_1, j_1, q_3, r_2, w_1, w_2, z_2, z_3, z_4]
    features = [year, month, day, weekday, hour] + topic_dummies
    
    return features


def extract_training_data_from_cases(case_list: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract training data from a list of completed cases during burn-in period.
    
    Args:
        case_list: List of case dictionaries from the burn-in period
        
    Returns:
        Tuple of (features_matrix, target_vector) where:
        - features_matrix: numpy array of shape (n_samples, n_features)
        - target_vector: numpy array of shape (n_samples,) with actual throughput times
    """
    features_list = []
    targets_list = []
    
    # Debug counters
    total_cases = len(case_list)
    burnin_cases = 0
    closed_cases = 0
    valid_cases = 0
    
    for case in case_list:
        # Check burn-in status
        is_burnin = case.get("burn_in", False)
        if is_burnin:
            burnin_cases += 1
        
        # Check closed status
        is_closed = case.get("status") == "closed"
        if is_closed:
            closed_cases += 1
            
        # Only use closed cases from burn-in period
        if is_burnin and is_closed:
            valid_cases += 1
            try:
                # Extract features
                features = extract_features_from_case(case)
                
                # Calculate actual throughput time (same as in simulation)
                if case["t_end"] and len(case["t_end"]) > 0:
                    actual_throughput_time = np.max(case["t_end"]) - case["q"]
                    
                    # Convert from days to minutes (for consistency with static model training)
                    actual_throughput_time_minutes = actual_throughput_time * 24 * 60
                    
                    features_list.append(features)
                    targets_list.append(actual_throughput_time_minutes)
                    
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(f"Skipping case {case.get('i', 'unknown')} due to missing data: {e}")
                continue
    
    # Log debug information
    logger.info(f"Training data extraction: Total cases={total_cases}, Burn-in cases={burnin_cases}, Closed cases={closed_cases}, Valid cases={valid_cases}, Final training samples={len(features_list)}")
    
    if len(features_list) == 0:
        logger.warning("No valid training data found in burn-in cases")
        return np.array([]), np.array([])
    
    features_matrix = np.array(features_list)
    target_vector = np.array(targets_list)
    
    logger.info(f"Extracted {len(features_list)} training samples from burn-in period")
    
    return features_matrix, target_vector


def exponential_regression_loss(params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """
    Loss function for exponential regression: y = exp(intercept + X @ beta)
    
    Args:
        params: Model parameters [intercept, beta_1, beta_2, ..., beta_n]
        X: Feature matrix
        y: Target values
        
    Returns:
        Mean squared error loss
    """
    intercept = params[0]
    betas = params[1:]
    
    # Avoid numerical overflow
    linear_combination = intercept + X @ betas
    linear_combination = np.clip(linear_combination, -700, 700)  # Prevent overflow
    
    predictions = np.exp(linear_combination)
    
    # Handle potential issues with extreme values
    predictions = np.clip(predictions, 1e-10, 1e10)
    y_clipped = np.clip(y, 1e-10, 1e10)
    
    # Mean squared error
    mse = np.mean((predictions - y_clipped) ** 2)
    
    return mse


def train_exponential_regression(X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Train an exponential regression model on the given data.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Target vector of shape (n_samples,)
        
    Returns:
        Dictionary containing model parameters and metadata
    """
    if len(X) == 0 or len(y) == 0:
        logger.error("Cannot train model with empty data")
        return None
    
    n_features = X.shape[1]
    
    # Convert to log-space for more stable optimization
    log_y = np.log(np.maximum(y, 1e-6))  # Avoid log(0)
    
    # Use simple linear regression in log space instead of complex exponential optimization
    try:
        # Add intercept column to X
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Solve normal equations: beta = (X'X)^(-1)X'y
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ log_y
        
        # Add small regularization for numerical stability
        regularization = 1e-6 * np.eye(XtX.shape[0])
        XtX_reg = XtX + regularization
        
        params = np.linalg.solve(XtX_reg, Xty)
        
        intercept = params[0]
        betas = params[1:].tolist()
        
        # Calculate model performance metrics on training data
        linear_combination = intercept + X @ np.array(betas)
        linear_combination = np.clip(linear_combination, -700, 700)
        predictions = np.exp(linear_combination)
        
        # Calculate MAE and MSE on training data
        mae_burnin = np.mean(np.abs(predictions - y))
        mse_burnin = np.mean((predictions - y) ** 2)
        final_loss = mse_burnin  # Use MSE as the loss
        n_samples = len(y)
        
        model_info = {
            "intercept": float(intercept),
            "betas": betas,
            "n_features": n_features,
            "n_training_samples": n_samples,
            "final_loss": float(final_loss),
            "mae_burnin": float(mae_burnin),
            "mse_burnin": float(mse_burnin),
            "training_successful": True,
            "feature_names": ["year", "month", "day", "weekday", "hour", 
                            "d_2", "g_1", "j_1", "q_3", "r_2", "w_1", "w_2", "z_2", "z_3", "z_4"]
        }
        
        logger.info(f"Model training successful. Loss: {final_loss:.4f}, MAE: {mae_burnin:.4f}, MSE: {mse_burnin:.4f}, Samples: {n_samples}")
        
    except np.linalg.LinAlgError as e:
        logger.error(f"Linear algebra error during model training: {e}")
        model_info = {"training_successful": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Exception during model training: {e}")
        model_info = {"training_successful": False, "error": str(e)}
    
    return model_info


def save_model(model_info: Dict, filepath: Path) -> None:
    """
    Save trained model parameters to JSON file.
    
    Args:
        model_info: Dictionary containing model parameters
        filepath: Path where to save the model JSON file
    """
    try:
        # Ensure directory exists
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
        
        if not model_info.get("training_successful", False):
            logger.warning("Loaded model was not successfully trained")
            return None
            
        logger.info(f"Model loaded from {filepath}")
        return model_info
        
    except Exception as e:
        logger.error(f"Failed to load model from {filepath}: {e}")
        return None


def predict_TT_dynamic(sigma: Dict, model_info: Optional[Dict] = None) -> Dict:
    """
    Predict throughput time using either dynamic trained model or static fallback.
    
    Args:
        sigma: Case dictionary containing case information
        model_info: Optional trained model parameters. If None, uses static model.
        
    Returns:
        Updated sigma dictionary with predicted throughput time
    """
    try:
        if model_info is not None and model_info.get("training_successful", False):
            # Use dynamic trained model
            features = extract_features_from_case(sigma)
            
            intercept = model_info["intercept"]
            betas = model_info["betas"]
            
            # Ensure feature-beta compatibility
            if len(features) != len(betas):
                logger.warning(f"Feature-beta length mismatch: {len(features)} vs {len(betas)}. Using static model.")
                return predict_TT_static_fallback(sigma)
            
            # Exponential regression prediction
            linear_combination = intercept + np.dot(features, betas)
            linear_combination = np.clip(linear_combination, -700, 700)  # Prevent overflow
            
            # Prediction in minutes
            y_minutes = np.exp(linear_combination)
            
            # Convert from minutes to days
            y_days = y_minutes / (60 * 24)
            
            # Update case with estimated throughput time
            sigma["est_throughputtime"] = y_days
            
            logger.debug(f"Dynamic prediction for case {sigma.get('i', 'unknown')}: {y_days:.4f} days")
            
        else:
            # Fallback to static model
            sigma = predict_TT_static_fallback(sigma)
            
    except Exception as e:
        logger.error(f"Error in dynamic prediction: {e}. Using static fallback.")
        sigma = predict_TT_static_fallback(sigma)
    
    return sigma


def predict_TT_static_fallback(sigma: Dict) -> Dict:
    """
    Fallback to static throughput time prediction (same as original predict_TT).
    
    Args:
        sigma: Case dictionary containing case information
        
    Returns:
        Updated sigma dictionary with predicted throughput time
    """
    # Import and use the original static prediction function
    from models.throughput import predict_TT
    return predict_TT(sigma)


def calculate_main_period_metrics(case_list: List[Dict], model_info: Dict, F_burn_in: int) -> Tuple[float, float, int]:
    """
    Calculate MAE and MSE for the main period cases using the trained dynamic model.
    
    Args:
        case_list: List of all cases from simulation
        model_info: Trained model information dictionary
        F_burn_in: Burn-in period threshold
        
    Returns:
        Tuple of (mae_main, mse_main, n_main_cases) where:
        - mae_main: Mean Absolute Error on main period cases
        - mse_main: Mean Squared Error on main period cases  
        - n_main_cases: Number of main period cases used for evaluation
    """
    if not model_info or not model_info.get("training_successful", False):
        logger.warning("No valid trained model provided for main period metrics calculation")
        return float('nan'), float('nan'), 0
    
    predictions_main = []
    actuals_main = []
    
    for case in case_list:
        # Only use closed cases from main period (after burn-in)
        if (not case.get("burn_in", True) and 
            case.get("status") == "closed" and 
            case.get("q", 0) >= F_burn_in):
            
            try:
                # Get the predicted throughput time (stored during simulation)
                predicted_tt = case.get("est_throughputtime", None)
                
                # Calculate actual throughput time
                if case["t_end"] and len(case["t_end"]) > 0 and predicted_tt is not None:
                    actual_tt = np.max(case["t_end"]) - case["q"]
                    
                    # Convert both to same units (days)
                    predictions_main.append(predicted_tt)
                    actuals_main.append(actual_tt)
                    
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(f"Skipping case {case.get('i', 'unknown')} in main period metrics: {e}")
                continue
    
    if len(predictions_main) == 0:
        logger.warning("No valid main period cases found for metrics calculation")
        return float('nan'), float('nan'), 0
    
    predictions_array = np.array(predictions_main)
    actuals_array = np.array(actuals_main)
    
    # Calculate MAE and MSE
    mae_main = np.mean(np.abs(predictions_array - actuals_array))
    mse_main = np.mean((predictions_array - actuals_array) ** 2)
    n_main_cases = len(predictions_main)
    
    logger.info(f"Main period metrics: MAE={mae_main:.4f}, MSE={mse_main:.4f}, N={n_main_cases}")
    
    return float(mae_main), float(mse_main), n_main_cases


def train_model_on_burn_in(case_list: List[Dict], run_dir: Path) -> Optional[Dict]:
    """
    Complete pipeline for training a model on burn-in data and saving it.
    
    Args:
        case_list: List of cases from burn-in period
        run_dir: Directory where to save the trained model
        
    Returns:
        Trained model info dictionary, or None if training failed
    """
    logger.info("Starting dynamic model training on burn-in data")
    
    # Extract training data
    X, y = extract_training_data_from_cases(case_list)
    
    if len(X) == 0:
        logger.warning("No training data available from burn-in period")
        return None
    
    # Train model
    model_info = train_exponential_regression(X, y)
    
    if model_info and model_info.get("training_successful", False):
        # Save model
        model_path = run_dir / "dynamic_throughput_model.json"
        save_model(model_info, model_path)
        return model_info
    else:
        logger.error("Model training failed")
        return None 