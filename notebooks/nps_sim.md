## NPS Simulation Model (`simulate_NPS`) Summary (Concise)

This model simulates the final NPS score for a case *after* its completion, using its actual throughput time, case topic, and sensitivity analysis parameters.

1.  **Objective**: Simulate a final NPS score reflecting actual case outcomes and incorporating sensitivity parameters (`F_NPS_dist_bias`, `F_tNPS_wtime_effect_bias`).

2.  **Core Inputs**:
    *   Actual Throughput Time (`y`): Converted to log-transformed minutes.
    *   Case Topic Index (`case_topicidx`): Used for one-hot encoding to apply topic-specific effects.
    *   `seed`: For reproducible random draws.
    *   Sensitivity Parameters: `F_NPS_dist_bias` and `F_tNPS_wtime_effect_bias`.

3.  **Model Type**: **Gamma Regression**.
    *   A linear predictor (`eta`) is calculated: `eta = biased_intercept + (biased_log_tt_coeff * log_throughputtime) + sum(topic_coeffs) `.
    *   The NPS is then drawn from a Gamma distribution: `NPS = random_gamma_draw(shape = exp(eta) / gammascale, scale = gammascale) - 1`.

4.  **Key Parameters & Sensitivity Adjustment**:
    *   **Intercept**: Base value (`2.300587`) is adjusted by adding `F_NPS_dist_bias`.
    *   **Log-Throughput Time Coefficient**: Base value (`-0.0098232`) is multiplied by `F_tNPS_wtime_effect_bias`.
    *   **Topic Coefficients**: A set of fixed coefficients for each case topic (e.g., d2: -0.12910, g1: 0.100756, etc.).
    *   **Gamma Scale (`gammascale`)**: Fixed value (`1.300057`), used in calculating the Gamma shape and as its scale parameter.

5.  **NPS Priority (`NPS_priority`)**: Calculated as `abs(simulated_NPS - 7.5)`.

6.  **Winsorizing**: The project context mentions winsorizing NPS to a 0-10 range. However, the code for this in the `simulate_NPS` function is currently commented out and thus **not active** in the direct script implementation.
