# Manipulating Predictability of Estimated Throughput Time

This document outlines an investigation into how the predictability of the *estimated throughput time* (`est_throughputtime`) for a case at its arrival can be manipulated within the simulation framework. This is analogous to how `F_NPS_dist_bias` and `F_tNPS_wtime_effect_bias` allow manipulation of the simulated NPS outcome.

## 1. Current Calculation of Estimated Throughput Time (`est_throughputtime`)

The initial estimation of throughput time for a new case occurs in the `CaseArrival` function within `src/NPS_SIM/algorithms/alg2_case_arrival.py`. This function calls `predict_TT_Notopic(sigma)` from `src/NPS_SIM/models/throughput.py`.

Analysis of `predict_TT_Notopic(sigma)` reveals the following:

*   **Deterministic Calculation:** The core logic for calculating `est_throughputtime` is:
    ```python
    # In src/NPS_SIM/models/throughput.py
    # ...
    betas = [-0.02950, #year
             -0.04149, #month
             -0.00034, #weekday
              0.06511] #hour
    X = [year, month, weekday, hour] # Derived from case arrival datetime
    c = 66.80872 # Intercept
    y = np.exp(c + np.dot(X,betas))-1 # Deterministic prediction
    # ...
    y = y/60/24 # Convert from minutes to days
    sigma["est_throughputtime"] = y
    ```
    This calculation is purely deterministic, based on the features of the case's arrival time (year, month, weekday, hour) and fixed model coefficients (`betas`, `c`).

*   **No Inherent Randomness (Currently):** As implemented, there is no random component applied to this estimation. The accuracy of the prediction is fixed by the model's formula and coefficients.

## 2. Potential for Introducing Tunable Predictability

Interestingly, the `predict_TT_Notopic` function in `src/NPS_SIM/models/throughput.py` contains a commented-out section of code:

```python
    #simulation expression (This section is commented out)
    #scale = 2.2944
    # y = np.random.normal(loc=linear_comb, # linear_comb is not defined here
    #                      scale=scale,
    #                      size=1)[0]
```

This suggests a prior intent or consideration to introduce a stochastic element to the estimation, likely by adding normally distributed noise around a central predicted value (`linear_comb`, which would correspond to the deterministic part `c + np.dot(X,betas)`). The `scale` parameter (2.2944) would represent the standard deviation of this noise.

## 3. Proposal for Implementing Tunable Predictability

To allow for the manipulation of `est_throughputtime` predictability, we can leverage and adapt the commented-out stochastic component. This would involve the following steps:

**A. Modify `src/NPS_SIM/models/throughput.py` (`predict_TT_Notopic` function):**

1.  **Introduce a New Parameter:** Add a new parameter to the function signature, for example, `est_TT_noise_scale_factor`, with a default value of `0.0` (or `1.0` if the base scale is always desired unless modified). A default of `0.0` would mean deterministic prediction by default, aligning with current behavior if the new factor isn't specified.
2.  **Define `linear_comb`:** Calculate the deterministic part of the prediction:
    `linear_comb = c + np.dot(X, betas)`
3.  **Reactivate Stochastic Component:** Uncomment the lines related to `np.random.normal`.
4.  **Apply the Scaling Factor:** Modify the scale of the normal distribution using the new parameter:
    `base_scale = 2.2944` (the original commented-out scale)
    `current_scale = base_scale * est_TT_noise_scale_factor`
5.  **Generate Prediction:**
    *   If `est_TT_noise_scale_factor` is `0` (or `current_scale` is `0`), then `y = np.exp(linear_comb) - 1`.
    *   Otherwise, `y_stochastic_component = np.random.normal(loc=linear_comb, scale=current_scale, size=1)[0]`, and then `y = np.exp(y_stochastic_component) - 1`. (The application of `np.exp` needs to be considered carefully depending on whether `linear_comb` is already log-transformed or not. The original model seems to be `y = np.exp(linear_comb)-1`, so the noise should ideally be added to `linear_comb` before exponentiation).

    A revised approach for applying noise to the linear combination before exponentiation:
    ```python
    linear_comb = c + np.dot(X, betas)
    base_scale = 2.2944 # This is the standard deviation for the noise on the linear combination
    
    if est_TT_noise_scale_factor > 0: # Only add noise if factor is positive
        noise = np.random.normal(loc=0, scale=(base_scale * est_TT_noise_scale_factor), size=1)[0]
        linear_comb_with_noise = linear_comb + noise
    else: # Deterministic
        linear_comb_with_noise = linear_comb
        
    y = np.exp(linear_comb_with_noise) - 1 
    ```

**B. Modify `src/NPS_SIM/algorithms/alg2_case_arrival.py`:**

1.  **Pass the New Factor:** When calling `predict_TT_Notopic`, pass the new simulation parameter (e.g., `F_est_TT_noise_scale`):
    `sigma = predict_TT_Notopic(sigma, est_TT_noise_scale_factor=settings["F_est_TT_noise_scale"])` (assuming `settings` contains this new factor).

**C. Update Experiment Configuration (`run_experiment.py`, `generate_design.py`):**

1.  **Add to `sim_params`:** In `src/NPS_SIM/run_experiment.py`, add the new factor (e.g., `F_est_TT_noise_scale`) to the `sim_params` dictionary, ensuring it's read from the experiment `settings` (which come from `design_table.csv`).
    ```python
    # Example in run_experiment.py
    sim_params = {
        # ... other params
        "F_est_TT_noise_scale": float(settings["F_est_TT_noise_scale"]),
        # ...
    }
    # This new parameter then needs to be passed down into CaseArrival,
    # and from there to predict_TT_Notopic.
    ```
2.  **Update Design Generation:** In `src/NPS_SIM/generate_design.py` (or the JSON settings file it uses), include this new factor so it can be varied across experiments.

**Impact of the `F_est_TT_noise_scale` factor:**

*   **`F_est_TT_noise_scale = 0`**: The prediction for `est_throughputtime` becomes purely deterministic (based on the model `np.exp(c + np.dot(X,betas))-1`). This makes the estimation "perfectly predictable" relative to the model's mean output.
*   **`F_est_TT_noise_scale = 1`**: The prediction includes a random error component with the `base_scale` (e.g., 2.2944 if that value is used from the commented code).
*   **`F_est_TT_noise_scale > 1`**: The variance of the estimation error increases, making the `est_throughputtime` "less predictable" (more noisy).
*   **`0 < F_est_TT_noise_scale < 1`**: The variance of the estimation error decreases, making the `est_throughputtime` "more predictable" (less noisy) around the model's mean output.

This approach provides a controllable way to study the impact of throughput time predictability on queueing performance and prioritization scheme effectiveness, particularly for schemes like SRTF and LRTF that rely on these estimates. 