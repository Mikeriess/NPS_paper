# Dynamic Throughput Time Prediction Models

## Overview

The dynamic throughput time (TT) prediction models are machine learning components that predict case resolution times based on case characteristics, temporal features, and real-time queue state information. These models are trained during the burn-in period of simulations and used to make predictions for cases arriving during the main simulation period.

**🚀 Latest Enhancements (December 2024)**: Feature scaling, cyclical temporal encoding, and optimized regularization for improved prediction accuracy.

## Purpose

- **Dynamic Prediction**: Improve upon static throughput time estimates by incorporating real-time system state
- **Queue Prioritization**: Enable NPS-based prioritization by providing accurate throughput time predictions
- **Research Analysis**: Support queue prioritization method comparisons in agent-based simulation studies

## Model Architecture

### Supported Model Types

1. **LASSO Regression** (`"Lasso"`) - **ENHANCED**
   - L1 regularization for feature selection
   - **Feature scaling** with StandardScaler for balanced regularization
   - **Reduced default regularization** (α = 0.01) for better performance
   - Handles multicollinearity well
   - Provides sparse solutions

2. **Gamma GLM** (`"gamma_glm"`)
   - Log-link Gamma regression
   - Appropriate for positive continuous outcomes (throughput times)
   - L2 regularization available
   - **Also enhanced** with feature scaling support

### Training Process

1. **Burn-in Period**: Model trains on completed cases during burn-in using FCFS prioritization
2. **Feature Extraction**: Convert case dictionaries to numerical feature vectors with **enhanced temporal encoding**
3. **Feature Scaling**: Apply StandardScaler normalization for balanced learning ✨ **NEW**
4. **Model Training**: Fit selected model type with optimized regularization
5. **Main Period**: Use trained model for dynamic predictions with automatic scaling application

## Feature Engineering

### 🎯 **December 2024 Major Enhancements**

#### **1. Feature Scaling Implementation**
- **StandardScaler normalization** applied to all features during training
- **Automatic scaling application** during prediction
- **Balanced regularization** across features with different scales
- **Saved scaling parameters** in model metadata for consistent inference

#### **2. Enhanced Temporal Feature Engineering**
- **Cyclical encoding** for month, day, and hour features
- **Year centering** around 2020 baseline to reduce scale
- **Log transformation** for highly skewed queue state features

#### **3. Optimized Regularization**
- **Reduced default α** from 0.1 to 0.01 for less aggressive feature selection
- **Better coefficient retention** leading to improved predictions
- **Enhanced model performance** with more informative features

### Previous Updates (June 2025)

**Major improvement**: Converted month, day of month and hour from one-hot encoding to numerical features.

#### Before vs After Feature Engineering

| Feature Type | June 2025 | December 2024 (Latest) | Improvement |
|--------------|-----------|------------------------|-------------|
| Month | 1 numerical | **2 cyclical** (sin, cos) | Captures seasonality |
| Day of month | 1 numerical | **2 cyclical** (sin, cos) | Captures monthly patterns |
| Hour | 1 numerical | **2 cyclical** (sin, cos) | Captures daily patterns |
| Year | Raw value (~2024) | **Centered** (year - 2020) | Reduced scale impact |
| Queue features | Raw values | **Log-transformed** (skewed) | Better distributions |
| **Total** | **34 features** | **37 features** | **Enhanced encoding** |

### Current Feature Structure (37 features) ✨ **UPDATED**

#### 1. Enhanced Temporal Features (7 features) ✨ **NEW**
- **Year (centered)**: 1 feature (year - 2020.0)
- **Month cyclical**: 2 features (`month_sin`, `month_cos`) - captures seasonal patterns
- **Day cyclical**: 2 features (`day_sin`, `day_cos`) - captures monthly patterns  
- **Hour cyclical**: 2 features (`hour_sin`, `hour_cos`) - captures daily patterns

#### 2. Weekday Features (7 features)
- **Weekday**: 7 one-hot features (`weekday_0` to `weekday_6`, Monday=0)

#### 3. Case Topic Features (10 features)
One-hot encoded case topics in specific order:
- `d_2`, `g_1`, `j_1`, `q_3`, `r_2`, `w_1`, `w_2`, `z_2`, `z_3`, `z_4`

#### 4. Enhanced Queue State Features (13 features) ✨ **IMPROVED**
Real-time system state with **log transformation** for skewed features:

**Agent Availability (3 features):**
- `qs_agents_available`: Number of available agents
- `qs_agents_busy`: Number of busy agents
- `qs_agent_utilization`: Fraction of agents currently busy

**Queue State (4 features):**
- `qs_queue_length`: Cases waiting in queue
- `qs_cases_in_process`: Cases currently being processed
- `qs_total_active_cases`: Total active cases in system
- `qs_queue_wait_time_current`: Current average queue waiting time **[LOG-TRANSFORMED]**

**System Throughput (4 features):**
- `qs_recent_completion_rate`: Recent case completion rate
- `qs_recent_avg_throughput_time`: Recent average throughput time **[LOG-TRANSFORMED]**
- `qs_recent_arrival_rate`: Recent case arrival rate
- `qs_workload_intensity`: System workload intensity measure

**Temporal Context (2 features):**
- `qs_time_since_last_completion`: Time since last case completion **[LOG-TRANSFORMED]**
- `qs_cases_arrived_today`: Number of cases arrived today **[LOG-TRANSFORMED]**

### 🎯 **Benefits of Enhanced Feature Engineering**

#### ✅ **Feature Scaling Advantages**
- **Balanced regularization**: All features treated equally regardless of original scale
- **Improved convergence**: Faster and more stable model training
- **Better coefficient interpretation**: Coefficients reflect true feature importance
- **Reduced overfitting**: More effective L1 penalty application

#### ✅ **Cyclical Temporal Encoding**
- **Periodicity capture**: Models understand that hour 23 is close to hour 0
- **Seasonal patterns**: Month encoding captures yearly cycles naturally
- **Continuity**: Smooth transitions between temporal boundaries
- **Improved generalization**: Better handling of temporal edge cases

#### ✅ **Optimized Regularization**
- **Less aggressive selection**: α = 0.01 vs 0.1 retains more informative features
- **Better predictions**: Reduced coefficient shrinkage improves accuracy
- **Balanced sparsity**: Still achieves feature selection without over-penalization

## Usage

### Basic Model Training ✨ **ENHANCED**

```python
from models.dynamic_throughput import train_model_on_burn_in

# Enhanced training with feature scaling and optimized regularization
model_info = train_model_on_burn_in(
    case_list=burn_in_cases,
    run_dir=results_directory,
    throughput_model_type="Lasso",  # or "gamma_glm"
    model_penalty_alpha=0.01        # Reduced from 0.1 for better performance
)

# Model info now includes scaling parameters:
# model_info["scaler_mean"], model_info["scaler_scale"], model_info["feature_scaling_applied"]
```

### Making Predictions ✨ **AUTO-SCALING**

```python
from models.dynamic_throughput import predict_TT_dynamic

# Case with queue state features - scaling applied automatically
case_dict = {
    'q_dt': datetime.datetime(2018, 7, 15, 14, 30),
    'c_topic': 2,  # Topic index
    'qs_agent_utilization': 0.75,
    'qs_queue_length': 5,
    # ... other queue state features
}

# Prediction automatically applies same scaling used during training
updated_case = predict_TT_dynamic(case_dict, model_info)
predicted_days = updated_case['est_throughputtime']
```

### Feature Extraction ✨ **ENHANCED**

```python
from models.dynamic_throughput import extract_features_from_case

features = extract_features_from_case(case_dict)
# Returns 37-dimensional feature vector with enhanced temporal encoding
# Features are automatically scaled during prediction when using trained model
```

## Implementation Details

### File Structure
- **`dynamic_throughput.py`**: Main module with enhanced training and prediction functions ✨ **UPDATED**
- **`lasso_tt.py`**: LASSO implementation (compatible with scaling)
- **`queue_state_features.py`**: Queue state feature calculation utilities
- **Model persistence**: JSON files with scaling parameters in `{run_dir}/models/dynamic_throughput_model.json`

### Enhanced Model Metadata ✨ **NEW**
```json
{
  "model_type": "lasso",
  "feature_scaling_applied": true,
  "scaler_mean": [...],  // StandardScaler mean values
  "scaler_scale": [...], // StandardScaler scale values
  "alpha_l1": 0.01,      // Reduced regularization
  "n_features": 37,      // Enhanced feature count
  // ... other model info
}
```

### Integration Points
- **Simulation**: `alg1_timeline_simulation.py` calls enhanced training at end of burn-in
- **Case arrival**: Queue state features calculated with log transformations during 15-minute intervals
- **Prioritization**: Dynamic predictions with automatic scaling used in NPS-based queue management

### Backward Compatibility ✨ **MAINTAINED**
- **Legacy models**: Non-scaled models automatically detected and handled
- **Feature versioning**: Robust handling of different feature set sizes (34 vs 37 features)
- **Static fallback**: Automatically falls back to static model if dynamic training fails
- **Missing features**: Missing queue state features default to 0.0

## Performance Improvements ✨ **METRICS**

### Expected Improvements from Enhancements

1. **Feature Scaling Impact**:
   - Reduced coefficient variance across features
   - More balanced feature selection
   - Improved numerical stability

2. **Cyclical Temporal Encoding**:
   - Better temporal pattern recognition
   - Reduced edge case prediction errors
   - Enhanced seasonal/daily trend capture

3. **Optimized Regularization**:
   - Retained informative features
   - Reduced prediction bias from over-regularization
   - Better model-data balance

### Performance Metrics

Models track the following enhanced metrics during training and evaluation:

- **MAE (Mean Absolute Error)**: Average prediction error in minutes
- **MSE (Mean Squared Error)**: Squared prediction error in minutes
- **Training samples**: Number of burn-in cases used for training
- **Feature scaling statistics**: Mean normalization and scale factors ✨ **NEW**
- **Feature count**: Verification of expected 37 features ✨ **UPDATED**

## Troubleshooting

### Common Issues ✨ **UPDATED**

1. **"No training data extracted"**: Insufficient completed cases in burn-in period
   - **Solution**: Increase burn-in period (`F_burn_in`) or case arrival rate

2. **"Feature dimension mismatch for scaling"**: Feature count inconsistency
   - **Solution**: Check that feature extraction returns 37 features ✨ **UPDATED**
   - **Legacy compatibility**: Older models (34 features) handled automatically

3. **Model training failed**: Linear algebra errors or convergence issues
   - **Solution**: Feature scaling should resolve most numerical stability issues ✨ **NEW**
   - **Alternative**: Adjust regularization parameter (`model_penalty_alpha`)

### Debug Information ✨ **ENHANCED**

Enable debug logging to see detailed information:
```python
import logging
logging.getLogger('models.dynamic_throughput').setLevel(logging.DEBUG)

# Enhanced logging now includes:
# - Feature scaling statistics
# - Temporal encoding verification  
# - Regularization effectiveness metrics
```

## Research Context

These enhanced models support the NPS-based queue prioritization research comparing:
- **FCFS** (First Come, First Served)
- **LRTF** (Longest Remaining Time First)  
- **SRTF** (Shortest Remaining Time First)
- **NPS** (Net Promoter Score-based prioritization)

The **enhanced dynamic models** enable significantly more accurate throughput time predictions for the NPS method, which prioritizes customers with previous passive NPS scores to potentially convert them to promoters.

## Version History

- **🚀 December 2024**: **MAJOR ENHANCEMENT** - Feature scaling, cyclical temporal encoding, optimized regularization
- **June 2025**: Numerical month/day/hour feature engineering (65.3% feature reduction)
- **Previous**: Queue state features integration (13 additional features)
- **Initial**: Basic temporal and case topic features

---

*For questions or issues with the enhanced dynamic throughput prediction models, refer to the main simulation documentation or research team.* 