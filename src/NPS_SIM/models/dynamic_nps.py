"""
Dynamic NPS Prediction Model

This module provides dynamic NPS prediction capabilities for the queue prioritization simulation.
It trains on burn-in period data and makes real-time predictions during the main simulation period.

Design follows the same patterns as dynamic_throughput.py for consistency.
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set up logging
logger = logging.getLogger(__name__)

def extract_features_from_case_for_nps(case_dict: Dict) -> List[float]:
    """
    Extract features from a case dictionary for NPS model training/prediction.
    Uses predicted throughput time and case topic as features.
    
    Args:
        case_dict: Dictionary containing case information with keys like 'est_throughputtime', 'c_topic'
        
    Returns:
        List of feature values.
    """
    
    # Get predicted throughput time (in days)
    throughput_time_days = case_dict.get("est_throughputtime", 0.0)
    
    # Convert to minutes and log-transform (same as static NPS model)
    throughput_time_minutes = throughput_time_days * 24 * 60
    log_throughput_time = np.log(1 + throughput_time_minutes)
    
    # Case topic one-hot encoding (10 features)
    # Order matches the static NPS model: d_2, g_1, j_1, q_3, r_2, w_1, w_2, z_2, z_3, z_4
    case_topicidx = case_dict.get("c_topic", 0)
    
    # This is the order of topics as they appear in the original full list from which case_topicidx is derived
    original_case_topics_list = ["j_1", "z_3", "q_3", "z_2", "r_2", "z_4", "d_2", "w_2", "g_1", "w_1"]
    case_topics_ordered_for_betas = ["d_2", "g_1", "j_1", "q_3", "r_2", "w_1", "w_2", "z_2", "z_3", "z_4"]
    
    current_casetopic_str = original_case_topics_list[case_topicidx]

    topic_dummies = [0.0] * 10
    try:
        idx_in_ordered_list = case_topics_ordered_for_betas.index(current_casetopic_str)
        topic_dummies[idx_in_ordered_list] = 1.0
    except ValueError:
        logger.warning(f"Case topic '{current_casetopic_str}' (index {case_topicidx}) not found in defined ordered list for dummies. Topic features will be all zero.")

    # Combine features: log_throughput_time (1) + topic_dummies (10) = 11 features total
    features = [log_throughput_time] + topic_dummies
    
    return features


def extract_training_data_from_cases_for_nps(case_list: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract training data from a list of completed cases during burn-in period for NPS prediction.
    
    Args:
        case_list: List of case dictionaries from the burn-in period
        
    Returns:
        Tuple of (features_matrix, target_vector) where:
        - features_matrix: numpy array of shape (n_samples, n_features)
        - target_vector: numpy array of shape (n_samples,) with simulated NPS scores
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
                features = extract_features_from_case_for_nps(case)
                
                # For NPS training, we need the actual simulated NPS as target
                # This would be calculated using the actual throughput time
                if case["t_end"] and len(case["t_end"]) > 0:
                    actual_throughput_time = np.max(case["t_end"]) - case["q"] # in days
                    
                    # Simulate the "true" NPS using the same approach as in store_evlog
                    from NPS_SIM.distributions.tNPS import simulate_NPS
                    
                    # Use a deterministic seed based on case ID for reproducible training
                    training_seed = hash(str(case.get('i', 0))) % (2**31)
                    
                    simulated_nps, _ = simulate_NPS(
                        case_topicidx=case.get("c_topic", 0),
                        y=actual_throughput_time,
                        seed=training_seed,
                        F_NPS_dist_bias=0,  # No bias during training
                        F_tNPS_wtime_effect_bias=1.0  # No bias during training
                    )
                    
                    features_list.append(features)
                    targets_list.append(simulated_nps)
                    
            except (KeyError, IndexError, TypeError) as e:
                logger.warning(f"Skipping case {case.get('i', 'unknown')} for NPS training due to missing data: {e}")
                continue
    
    logger.info(f"NPS training data extraction: Total cases={total_cases}, Burn-in cases={burnin_cases}, Closed cases={closed_cases}, Valid cases={valid_cases}, Final training samples={len(features_list)}")
    
    if len(features_list) == 0:
        logger.warning("No valid training data found in burn-in cases for dynamic NPS model.")
        return np.array([]).reshape(0,0), np.array([])

    features_matrix = np.array(features_list)
    target_vector = np.array(targets_list)
    
    logger.info(f"Extracted {len(features_list)} training samples from burn-in period for dynamic NPS model.")
    
    return features_matrix, target_vector


def predict_with_lasso_nps(features: np.ndarray, model_info: Dict) -> Optional[float]:
    """
    Make NPS prediction using trained Lasso model.
    
    Args:
        features: Feature vector for a single case
        model_info: Dictionary containing trained model information
        
    Returns:
        Predicted NPS score or None if prediction fails
    """
    try:
        coefficients = np.array(model_info["coefficients"])
        intercept = model_info["intercept"]
        
        if len(features) != len(coefficients):
            logger.warning(f"Feature dimension mismatch for NPS prediction: expected {len(coefficients)}, got {len(features)}")
            return None
            
        prediction = intercept + np.dot(features, coefficients)
        return float(prediction)
        
    except Exception as e:
        logger.warning(f"Error in Lasso NPS prediction: {e}")
        return None


def predict_with_gamma_nps(features: np.ndarray, model_info: Dict) -> Optional[float]:
    """
    Make NPS prediction using trained Gamma GLM model.
    
    Args:
        features: Feature vector for a single case
        model_info: Dictionary containing trained model information
        
    Returns:
        Predicted NPS score or None if prediction fails
    """
    try:
        coefficients = np.array(model_info["coefficients"])
        intercept = model_info["intercept"]
        
        if len(features) != len(coefficients):
            logger.warning(f"Feature dimension mismatch for NPS Gamma prediction: expected {len(coefficients)}, got {len(features)}")
            return None
            
        # Gamma GLM with log link
        linear_predictor = intercept + np.dot(features, coefficients)
        prediction = np.exp(linear_predictor)
        
        return float(prediction)
        
    except Exception as e:
        logger.warning(f"Error in Gamma NPS prediction: {e}")
        return None


def predict_NPS_dynamic(case_dict: Dict, model_info: Dict) -> Dict:
    """
    Make dynamic NPS prediction for a case using trained model.
    
    Args:
        case_dict: Case dictionary containing features
        model_info: Trained model information
        
    Returns:
        Updated case dictionary with dynamic NPS prediction
    """
    try:
        features = np.array(extract_features_from_case_for_nps(case_dict))
        
        model_type = model_info.get("model_type", "lasso")
        
        if model_type == "lasso":
            predicted_nps = predict_with_lasso_nps(features, model_info)
        elif model_type == "gamma_glm":
            predicted_nps = predict_with_gamma_nps(features, model_info)
        else:
            logger.warning(f"Unknown NPS model type: {model_type}")
            return case_dict
            
        if predicted_nps is not None:
            # Update case with dynamic NPS prediction
            updated_case = case_dict.copy()
            updated_case["est_NPS"] = predicted_nps
            updated_case["est_NPS_priority"] = abs(predicted_nps - 7.5)
            
            return updated_case
        else:
            logger.warning("Dynamic NPS prediction failed, keeping original prediction")
            return case_dict
            
    except Exception as e:
        logger.warning(f"Error in dynamic NPS prediction: {e}")
        return case_dict


def calculate_main_period_metrics_nps(case_list: List[Dict], model_info: Dict, F_burn_in: int) -> Tuple[float, float, int]:
    """
    Calculate MAE and MSE for the main simulation period using a trained dynamic NPS model.
    Uses the simulated_NPS values that were already calculated and stored during the simulation.
    """
    main_period_predictions = []
    main_period_actuals = []
    
    model_type = model_info.get("model_type", "unknown")

    for case in case_list:
        if not case.get("burn_in", True) and case.get("status") == "closed": # Main period, closed cases
            try:
                features = np.array(extract_features_from_case_for_nps(case))
                
                prediction: Optional[float] = None
                if model_type == "lasso":
                    prediction = predict_with_lasso_nps(features, model_info)
                elif model_type == "gamma_glm":
                    prediction = predict_with_gamma_nps(features, model_info)
                else:
                    logger.warning(f"Cannot calculate main period NPS metrics for unknown model type: {model_type}")
                    continue

                # Use the simulated_NPS that was already calculated during simulation
                # This should be available in the case dictionary as it's calculated in store_evlog
                simulated_nps = case.get("simulated_NPS")
                
                if prediction is not None and simulated_nps is not None:
                    main_period_predictions.append(prediction)
                    main_period_actuals.append(simulated_nps)
                else:
                    if prediction is None:
                        logger.warning(f"No prediction available for case {case.get('i', 'unknown')}")
                    if simulated_nps is None:
                        logger.warning(f"No simulated_NPS available for case {case.get('i', 'unknown')}")
                        
            except Exception as e:
                logger.warning(f"Skipping case for main period NPS metrics due to error: {e}")
                continue

    if not main_period_predictions or not main_period_actuals:
        logger.info("No valid cases found for calculating main period NPS model metrics.")
        return float('nan'), float('nan'), 0

    mae_main = mean_absolute_error(main_period_actuals, main_period_predictions)
    mse_main = mean_squared_error(main_period_actuals, main_period_predictions)
    n_cases = len(main_period_predictions)
    
    return mae_main, mse_main, n_cases


def train_nps_model_on_burn_in(
    case_list: List[Dict], 
    run_dir: Path,
    nps_model_type: str = "Lasso",
    model_penalty_alpha: float = 0.1
) -> Optional[Dict]:
    """
    Train an NPS prediction model using data from the burn-in period.
    Saves the model_info to a JSON file in the run_dir/models.
    """
    logger.info(f"Starting dynamic NPS model training ({nps_model_type}) on burn-in data with penalty alpha={model_penalty_alpha}...")
    
    X_train, y_train = extract_training_data_from_cases_for_nps(case_list)
    
    if X_train.shape[0] == 0:
        logger.warning("No training data extracted. Dynamic NPS model training aborted.")
        return {"training_successful": False, "error": "No training data from burn-in"}

    # Define feature names based on the extract_features_from_case_for_nps logic
    # Log throughput time (1) + Topics (10) = 11 features
    feature_names = ['log_throughput_time'] + \
                    ["d_2", "g_1", "j_1", "q_3", "r_2", "w_1", "w_2", "z_2", "z_3", "z_4"]

    model_info: Optional[Dict] = None
    
    try:
        if nps_model_type.lower() == "lasso":
            # Train Lasso regression
            model = Lasso(alpha=model_penalty_alpha, random_state=42)
            model.fit(X_train, y_train)
            
            # Calculate training metrics
            y_pred_train = model.predict(X_train)
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mse_train = mean_squared_error(y_train, y_pred_train)
            
            model_info = {
                "model_type": "lasso",
                "coefficients": model.coef_.tolist(),
                "intercept": float(model.intercept_),
                "feature_names": feature_names,
                "n_features": len(feature_names),
                "n_training_samples": X_train.shape[0],
                "training_successful": True,
                "mae_train": mae_train,
                "mse_train": mse_train,
                "penalty_alpha": model_penalty_alpha
            }
            
            logger.info(f"Lasso NPS model trained successfully. Training MAE: {mae_train:.4f}, MSE: {mse_train:.4f}")
            
        elif nps_model_type.lower() == "gamma_glm":
            # For Gamma GLM, we'll use a simple implementation with log link
            # This is a simplified version - in practice you might want to use statsmodels
            from sklearn.linear_model import Ridge
            
            # Use log of target for Gamma-like behavior (ensuring positive targets)
            y_train_log = np.log(np.maximum(y_train + 10, 0.1))  # Shift to ensure positive values
            
            model = Ridge(alpha=model_penalty_alpha, random_state=42)
            model.fit(X_train, y_train_log)
            
            # Calculate training metrics (transform back)
            y_pred_train_log = model.predict(X_train)
            y_pred_train = np.exp(y_pred_train_log) - 10  # Transform back
            
            mae_train = mean_absolute_error(y_train, y_pred_train)
            mse_train = mean_squared_error(y_train, y_pred_train)
            
            model_info = {
                "model_type": "gamma_glm",
                "coefficients": model.coef_.tolist(),
                "intercept": float(model.intercept_),
                "feature_names": feature_names,
                "n_features": len(feature_names),
                "n_training_samples": X_train.shape[0],
                "training_successful": True,
                "mae_train": mae_train,
                "mse_train": mse_train,
                "penalty_alpha": model_penalty_alpha
            }
            
            logger.info(f"Gamma GLM NPS model trained successfully. Training MAE: {mae_train:.4f}, MSE: {mse_train:.4f}")
            
        else:
            logger.error(f"Unknown NPS model type: {nps_model_type}")
            return {"training_successful": False, "error": f"Unknown model type: {nps_model_type}"}
            
    except Exception as e:
        logger.error(f"Error during NPS model training: {e}")
        return {"training_successful": False, "error": str(e)}
    
    # Save model to file
    try:
        models_dir = run_dir / "models"
        models_dir.mkdir(exist_ok=True)
        
        model_file = models_dir / "dynamic_nps_model.json"
        with open(model_file, 'w') as f:
            json.dump(model_info, f, indent=2)
            
        logger.info(f"Dynamic NPS model saved to: {model_file}")
        
    except Exception as e:
        logger.warning(f"Failed to save NPS model to file: {e}")
        # Continue anyway, model_info is still valid for this run
    
    return model_info


def load_nps_model(model_file: Path) -> Optional[Dict]:
    """
    Load a trained NPS model from JSON file.
    
    Args:
        model_file: Path to the model JSON file
        
    Returns:
        Model info dictionary or None if loading fails
    """
    try:
        with open(model_file, 'r') as f:
            model_info = json.load(f)
        
        logger.info(f"Dynamic NPS model loaded from: {model_file}")
        return model_info
        
    except Exception as e:
        logger.warning(f"Failed to load NPS model from {model_file}: {e}")
        return None 