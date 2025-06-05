## NPS Prediction Model (`predict_NPS`) Summary

The simulation predicts an "estimated NPS" (`est_NPS`) for cases, which then informs an "NPS priority".

1.  **Objective**: Predict `est_NPS` based on case characteristics, primarily throughput time and case topic.

2.  **Input Features**:
    *   **Estimated Throughput Time (`est_throughputtime`)**: Converted from days to minutes, then log-transformed: `log(1 + throughput_time_minutes)`.
    *   **Case Topic (`c_topic`)**: An index used to select the specific topic for the case, which is then one-hot encoded into binary features.

3.  **Model Type**: A linear model whose output is exponentiated and scaled. This structure is akin to a Generalized Linear Model (GLM), potentially related to Gamma regression.

4.  **Prediction Equation for `est_NPS`**:
    `est_NPS = (exp(intercept + (beta_tt * log_throughputtime) + sum(beta_topic_k * topic_k_active)) / gammascale) - 1`
    *   `intercept`: Fixed value (`2.300587`).
    *   `beta_tt`: Coefficient for log-transformed throughput time (`-0.0098232`).
    *   `beta_topic_k`: Coefficients for each one-hot encoded case topic (various hardcoded values, some are 0).
    *   `topic_k_active`: Binary (0/1) indicating if a specific topic `k` is active for the case.
    *   `gammascale`: Fixed scaling factor (`1.300057`).

5.  **NPS Priority (`est_NPS_priority`)**: Calculated as `abs(est_NPS - 7.5)`, prioritizing cases with predicted NPS scores further from a neutral 7.5.

6.  **Process**: The model takes case data, transforms inputs, applies the formula with hardcoded parameters, and updates the case data with `est_NPS` and `est_NPS_priority`.

A simpler function, `predict_NPS_Notopic`, exists that excludes case topics and uses a simpler linear prediction without exponentiation or scaling.
