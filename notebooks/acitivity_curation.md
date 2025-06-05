### Activity Duration Model

Activity durations are randomly sampled from a **Weibull distribution**. The parameters of this distribution are determined as follows:

1.  **Shape Parameter (k)**: This is dynamically calculated as `k = exp(linear_combination)`,
    where `linear_combination` is:
    `intercept + case_topic_effect + personality_effect + activity_effect + (task_number_effect * task_number)`

2.  **Scale Parameter (λ)**: This is a fixed value.

**Coefficients and Parameters:**

*   **For the `linear_combination` (determining the shape `k`):**
    *   `intercept`: 1.6645
    *   `task_number_effect` (coefficient for `task_number`): 0.0420
    *   `personality_effect`: The agent's personality score (coefficient is 1.0).
    *   `case_topic_effect` (selected based on the case's topic index):
        *   Topic 0: -0.0557
        *   Topic 1: 0.1637
        *   Topic 2: 0.1712
        *   Topic 3: -0.0420
        *   Topic 4: 0.0836
        *   Topic 5: 0.0000 (Reference)
        *   Topic 6: 0.0200
        *   Topic 7: 0.0119
        *   Topic 8: -0.0538
        *   Topic 9: -0.0609
    *   `activity_effect` (selected based on the current activity type):
        *   "Interaction": 0.1057
        *   "Email": 0.0180
        *   "Task-Reminder": 0.0 (Reference)
        *   "END": 0.0 (Not applicable as END has no duration)

*   **For the Weibull distribution itself:**
    *   `scale_lambda` (the fixed scale parameter λ, `scale_theta` in code): 0.3908

The case topic used for the `case_topic_effect` is assigned from pre-generated `p_vectors` when the case processing begins with an agent.