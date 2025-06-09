# Dynamic NPS Prediction Model

This module provides dynamic NPS prediction capabilities for the queue prioritization simulation research. It follows the same design patterns as the dynamic throughput model for consistency and integration.

## Overview

The dynamic NPS model trains on burn-in period data and makes real-time predictions during the main simulation period, enabling more accurate NPS-based queue prioritization compared to the static NPS model.

### Key Features

- **Real-time Learning**: Trains on actual burn-in case outcomes
- **Feature Integration**: Uses predicted throughput time + case topic as features
- **Multiple Algorithms**: Supports Lasso regression and Gamma GLM
- **Performance Tracking**: Comprehensive metrics for model evaluation
- **Seamless Integration**: Follows same patterns as dynamic throughput model

## Model Architecture

### Supported Model Types

1. **LASSO Regression** (`"Lasso"`)
   - L1 regularization for feature selection
   - Linear relationship between features and NPS
   - Robust to multicollinearity

2. **Gamma GLM** (`"Gamma_GLM"`)
   - Log-link Gamma regression approach
   - Handles positive continuous NPS outcomes
   - L2 regularization available

### Training Process

1. **Burn-in Period**: Model trains on completed cases during burn-in using actual simulated NPS scores
2. **Feature Extraction**: Convert case dictionaries to 11-dimensional feature vectors
3. **Model Training**: Fit selected model type with specified regularization
4. **Main Period**: Use trained model for dynamic NPS predictions on new cases

## Feature Engineering

### Feature Structure (11 features)

#### 1. Throughput Time Feature (1 feature)
- **Log Throughput Time**: `log(1 + predicted_throughput_time_minutes)`
  - Uses the predicted throughput time from the throughput model
  - Log-transformed to match static NPS model structure
  - Captures the relationship between service time and customer satisfaction

#### 2. Case Topic Features (10 features)
One-hot encoded case topics in specific order matching static NPS model:
- `d_2`, `g_1`, `j_1`, `q_3`, `r_2`, `w_1`, `w_2`, `z_2`, `z_3`, `z_4`

### Benefits of This Feature Set

#### ✅ **Consistency with Static Model**
- Same feature transformation as static NPS prediction
- Maintains interpretability and comparability
- Leverages existing domain knowledge

#### ✅ **Simplicity and Efficiency**
- Only 11 features vs 34 for throughput model
- Fast training and prediction
- Reduced overfitting risk

#### ✅ **Logical Dependencies**
- NPS prediction depends on throughput prediction
- Captures the core hypothesis: throughput time affects customer satisfaction
- Case topic provides context-specific adjustments

## Usage

### Basic Model Training

```python
from models.dynamic_nps import train_nps_model_on_burn_in

nps_model_info = train_nps_model_on_burn_in(
    case_list=burn_in_cases,
    run_dir=results_directory,
    nps_model_type="Lasso",  # or "Gamma_GLM"
    model_penalty_alpha=0.1
)
```

### Making Predictions

```python
from models.dynamic_nps import predict_NPS_dynamic

# Case with predicted throughput time
case_dict = {
    'est_throughputtime': 2.5,  # Predicted throughput time in days
    'c_topic': 2,  # Topic index
    'est_NPS': 6.5,  # Original static prediction
    'est_NPS_priority': 1.0
}

updated_case = predict_NPS_dynamic(case_dict, nps_model_info)
predicted_nps = updated_case['est_NPS']
predicted_priority = updated_case['est_NPS_priority']
```

### Feature Extraction

```python
from models.dynamic_nps import extract_features_from_case_for_nps

features = extract_features_from_case_for_nps(case_dict)
# Returns 11-dimensional feature vector
```

## Implementation Details

### File Structure
- **`dynamic_nps.py`**: Main module with training and prediction functions
- **Model persistence**: JSON files saved in `{run_dir}/models/dynamic_nps_model.json`

### Integration Points
- **Simulation**: `alg1_timeline_simulation.py` calls training at end of burn-in
- **Case arrival**: Dynamic predictions made during 15-minute intervals
- **Prioritization**: Dynamic NPS predictions used in NPS-based queue management
- **Dependency**: Requires throughput predictions to be updated first

### Training Data Generation
- Uses `simulate_NPS()` function to generate "true" NPS scores for training
- Based on actual throughput times from burn-in period
- Deterministic seeds ensure reproducible training data

### Backward Compatibility
- **Static fallback**: Automatically falls back to static model if dynamic training fails
- **Feature handling**: Robust handling of missing features
- **Error tolerance**: Graceful degradation when predictions fail

## Performance Metrics

Models track the following metrics during training and evaluation:

- **MAE (Mean Absolute Error)**: Average prediction error in NPS points
- **MSE (Mean Squared Error)**: Squared prediction error in NPS points
- **Training samples**: Number of burn-in cases used for training
- **Feature count**: Verification of expected 11 features

## Integration with Simulation

### Timeline Integration

1. **Case Generation** (`CaseArrival`):
   ```python
   sigma = predict_TT(sigma)      # First predict throughput time
   sigma = predict_NPS(sigma)     # Then predict NPS (static)
   ```

2. **Burn-in Training** (End of burn-in period):
   ```python
   # Train both models
   dynamic_model_info = train_model_on_burn_in(...)           # Throughput
   dynamic_nps_model_info = train_nps_model_on_burn_in(...)   # NPS
   ```

3. **Real-time Updates** (15-minute intervals):
   ```python
   # Update predictions for newly arrived cases
   updated_case = predict_TT_dynamic(case, dynamic_model_info)      # First
   updated_case = predict_NPS_dynamic(case, dynamic_nps_model_info) # Then
   ```

### Parameter Integration

The simulation function now accepts NPS model parameters:

```python
Run_simulation(
    # ... existing parameters ...
    F_nps_model="Lasso",           # "Static", "Lasso", "Gamma_GLM"
    F_nps_model_penalty=0.1        # Regularization parameter
)
```

## Research Context

### Purpose in Queue Prioritization Study

1. **Enhanced NPS-based Prioritization**: More accurate NPS predictions lead to better identification of passive customers who could benefit from priority treatment

2. **Dynamic Learning**: Model adapts to actual system performance during burn-in, capturing real relationships between throughput time and customer satisfaction

3. **Method Comparison**: Enables fair comparison between static and dynamic approaches for NPS-based prioritization against FCFS, LRTF, and SRTF

### Key Research Questions

- **Prediction Accuracy**: How much does dynamic learning improve NPS prediction accuracy?
- **Queue Performance**: Do better NPS predictions lead to better queue management outcomes?
- **Sensitivity Analysis**: How robust are the results to different model types and parameters?

## Troubleshooting

### Common Issues

1. **"No training data extracted"**: Insufficient completed cases in burn-in period
   - **Solution**: Increase burn-in period (`F_burn_in`) or case arrival rate

2. **Model training failed**: Linear algebra errors or convergence issues
   - **Solution**: Adjust regularization parameter (`F_nps_model_penalty`)

3. **Feature count mismatch**: Expected 11 features but got different count
   - **Solution**: Check throughput prediction and case topic format

4. **Import errors**: Module not found for `NPS_SIM.distributions.tNPS`
   - **Solution**: Ensure proper Python path setup and module structure

### Debug Information

Enable debug logging to see detailed information:
```python
import logging
logging.getLogger('models.dynamic_nps').setLevel(logging.DEBUG)
```

## Comparison: Static vs Dynamic NPS Models

| Aspect | Static NPS Model | Dynamic NPS Model |
|--------|------------------|-------------------|
| **Training Data** | Pre-defined coefficients | Burn-in period cases |
| **Features** | Log throughput + topics | Log throughput + topics |
| **Adaptability** | Fixed parameters | Learns from actual data |
| **Prediction Timing** | Case arrival | Real-time during simulation |
| **Performance Tracking** | None | MAE, MSE, sample counts |
| **Model Types** | Single approach | Lasso, Gamma GLM options |
| **Integration** | Built into case arrival | Separate training + prediction |

## Version History

- **Initial**: Dynamic NPS model implementation following dynamic throughput patterns
- **Features**: 11-feature design (log throughput + 10 topics)
- **Integration**: Full simulation timeline integration with performance metrics

---

*For questions or issues with the dynamic NPS prediction model, refer to the main simulation documentation or research team.* 