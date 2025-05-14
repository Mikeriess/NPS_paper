# Simulation Metrics Documentation

This document outlines the key metrics generated and analyzed in the Queue Prioritization NPS Simulation project. Understanding these metrics is crucial for interpreting simulation results and the impact of different prioritization strategies.

The metrics flow through the following stages:
1.  **Base Data Generation:** Raw event data and case attributes are produced by the simulation algorithms (primarily in `src/NPS_SIM/algorithms/alg1_timeline_simulation.py` and supporting algorithm files). This includes detailed event logs (`evlog`) and a case database (`Case_DB`).
2.  **Per-Run Aggregation:** The `src/NPS_SIM/run_experiment.py` script processes the raw `evlog` and `Case_DB` for each simulation run. It calculates various aggregate statistics (means, counts) and stores them in `design_table.csv`.
3.  **Reporting and Further Analysis:** The `analysis/report_from_results.py` script reads `design_table.csv` to generate PDF reports, visualizations, and may calculate additional derived metrics for presentation.

## 0. Simulation Design Factors (from `design_table.csv`)

These factors are the input parameters varied for each simulation run, as defined in the `design_table.csv`. They control the conditions and configuration of the simulation.

*   **`F_number_of_agents`**:
    *   **Description:** Specifies the total number of agents available in the simulated contact center to process cases.
    *   **Significance:** Directly impacts system capacity, queue lengths, and waiting times.

*   **`F_priority_scheme`**:
    *   **Description:** Defines the rule used to order cases in the queue when an agent becomes available.
    *   **Examples:** `FCFS` (First-Come, First-Served), `SRTF` (Shortest Remaining Time First), `LRTF` (Longest Remaining Time First), `NPS` (NPS-based prioritization).
    *   **Significance:** A core experimental variable for comparing different operational strategies.

*   **`F_hard_ceiling`**:
    *   **Description:** Indicates the type of hard service level agreement (SLA) ceiling applied to cases in the queue. Typically, this might be "NO" (no ceiling) or "SLA" (a time-based ceiling is active).
    *   **Significance:** Determines if extremely long-waiting cases are escalated or given highest priority.

*   **`F_ceiling_value`**:
    *   **Description:** If `F_hard_ceiling` is active (e.g., "SLA"), this value specifies the time threshold (in days) beyond which a case in the queue receives overriding priority.
    *   **Significance:** Quantifies the threshold for the hard ceiling mechanism.

*   **`F_days`**:
    *   **Description:** The total duration for which the simulation is run, in days.
    *   **Significance:** Defines the observation window for the simulation.

*   **`F_burn_in`**:
    *   **Description:** The initial number of days of the simulation considered as a "burn-in" or warm-up period. Data from this period might be excluded from final analysis to ensure the system has reached a more stable state.
    *   **Significance:** Helps in obtaining more representative steady-state results by discarding initial transient effects.

*   **`F_NPS_dist_bias`**:
    *   **Description:** An additive bias term that is applied to the intercept (alpha) of the Gamma regression model used to simulate NPS scores from throughput times.
    *   **Significance:** One of the key manipulation variables for Study 3, used to investigate the sensitivity of NPS outcomes to shifts in the underlying NPS distribution.

*   **`F_tNPS_wtime_effect_bias`**:
    *   **Description:** A multiplicative scaling factor that is applied to the log-transformed throughput time effect (beta) in the Gamma regression model used for simulating NPS scores.
    *   **Significance:** The second key manipulation variable for Study 3, used to explore the sensitivity of NPS outcomes to changes in the strength of the relationship between throughput time and NPS.

*   **`seed`** (often corresponds to the `run` number):
    *   **Description:** The random seed value used for a specific simulation run.
    *   **Significance:** Ensures reproducibility of the simulation results for a given set of factor levels. Different seeds allow for exploring variability due to stochastic elements.

*   **`startdate`**:
    *   **Description:** The calendar date on which the simulation period begins.
    *   **Significance:** Provides a temporal anchor for the simulation, relevant for time-dependent effects like seasonality in arrivals if modeled.

## I. Base Metrics (Raw data from simulation, recorded in `evlog` and `Case_DB`)

These are fundamental data points recorded during or immediately after the simulation of each case.

*   **`simulated_throughput_time`** (Column in `evlog`):
    *   **Definition:** The actual total time a case spent in the system, from its arrival to its final closure.
    *   **Calculation:** `max(case_activity_end_times) - case_arrival_time`.
    *   **Unit:** Typically in days or fractions of days, but the report script might convert it to hours for display.
    *   **Significance:** A primary performance indicator; directly influences the `simulated_NPS`.

*   **`simulated_NPS`** (Column in `evlog`):
    *   **Definition:** The Net Promoter Score calculated for a case *after* it has been closed, based on its actual `simulated_throughput_time`.
    *   **Calculation:** Derived from a Gamma regression model: `NPS ~ f(throughput_time, case_topic)`. The model incorporates:
        *   Log-transformed `simulated_throughput_time`.
        *   `case_topic`.
        *   `F_NPS_dist_bias`: An additive bias to the intercept of the Gamma regression.
        *   `F_tNPS_wtime_effect_bias`: A multiplicative scaling factor for the log-transformed throughput time effect.
        The output is then winsorized to the standard NPS range (-100 to 100).
    *   **Significance:** Represents the "true" customer satisfaction outcome for a simulated case according to the model. A key metric for comparing prioritization strategies, especially the NPS-based one.

*   **`simulated_NPS_priority`** (Column in `evlog`):
    *   **Definition:** A priority score associated with the `simulated_NPS`. Its exact calculation (e.g., if it's different from `simulated_NPS` itself or a transformed version) is determined by the `simulate_NPS` function in `distributions.tNPS`.
    *   **Significance:** Likely used for internal consistency checks or if the raw NPS score undergoes a transformation before being used as a priority value in some contexts, though `est_NPS_priority` is more directly used for queue ordering in the NPS scheme.

*   **`est_throughput_time`** (Column `est_throughputtime` in `evlog` from simulation, mapped to `est_throughput_time` in `run_experiment.py`):
    *   **Definition:** An *estimated* throughput time for a case. This prediction is made during the simulation (e.g., at case arrival).
    *   **Calculation:** Based on predictive models (e.g., `predict_TT_Notopic` in `alg2_case_arrival.py`).
    *   **Significance:** Used by some prioritization schemes (like SRTF, LRTF) and as an input for `est_NPS`.

*   **`est_NPS`** (Column in `evlog`):
    *   **Definition:** An *estimated* NPS value for a case, predicted during the simulation, likely using `est_throughput_time`.
    *   **Calculation:** Based on predictive models (e.g., `predict_NPS_Notopic` in `alg2_case_arrival.py`).
    *   **Significance:** Used by the NPS-based prioritization method to rank cases in the queue.

*   **`est_NPS_priority`** (Column in `evlog`):
    *   **Definition:** The priority score derived from `est_NPS` (or related estimations) that is actively used by the NPS prioritization scheme to order cases in the queue. Higher values typically mean higher priority.
    *   **Significance:** The direct value used for making prioritization decisions in the NPS scheme.

*   **`initial_delay`** (Column in `evlog`):
    *   **Definition:** The time a case waits from its arrival in the system until its first processing activity begins (i.e., when it's assigned to an agent and work commences).
    *   **Significance:** Measures immediate responsiveness of the system.

*   **`case_status`** (Column in `evlog`):
    *   **Definition:** The status of a case at the time a particular event in the `evlog` is recorded (e.g., "open", "active", "queued", "closed").
    *   **Significance:** Allows filtering of events, for example, to analyze only closed cases.

*   **`status`** (Column in `Case_DB`):
    *   **Definition:** The final status of a case at the end of the simulation run (e.g., "closed", "open_at_end").
    *   **Significance:** Used to count total closed cases.

*   **`case_queued`** (Column in `Case_DB`):
    *   **Definition:** A boolean flag indicating whether a particular case was ever placed in a queue during its lifecycle within the simulation.
    *   **Significance:** Used to count how many cases experienced queuing.

## II. Aggregated Metrics (Calculated per run by `run_experiment.py`, stored in `design_table.csv`)

These metrics are calculated at the end of each simulation run by `run_experiment.py`, summarizing the performance of that specific configuration. The `evlog` stores `simulated_NPS`, `simulated_throughput_time`, etc., for each *event* of a case. When `run_experiment.py` calculates means from `evlog` directly (e.g. `evlog.simulated_NPS.mean()`), it's averaging over all events. However, since `simulated_NPS` and `simulated_throughput_time` are typically constant for all events of a single closed case and only defined upon closure, the `all_avg_*` versions for these metrics effectively reflect the average over closed cases, but this depends on how NaNs for non-closed cases are handled by `.mean()`.

### A. Metrics Averaged Over **Closed** Cases
These metrics are calculated by first filtering the `evlog` for events belonging to cases with `case_status == "closed"`.

*   **`closed_avg_simulated_NPS`**:
    *   **Definition:** The average of `simulated_NPS` values across all cases that were *closed* during a single simulation run.
    *   **Calculation in `run_experiment.py`:** `evlog[evlog.case_status == "closed"].simulated_NPS.mean()`

*   **`closed_avg_simulated_throughput_time`**:
    *   **Definition:** The average of `simulated_throughput_time` values across all *closed* cases in a run.
    *   **Calculation in `run_experiment.py`:** `evlog[evlog.case_status == "closed"].simulated_throughput_time.mean()`

*   **`closed_avg_predicted_NPS`**:
    *   **Definition:** The average of `est_NPS` values (predictions made during simulation) for those cases that were eventually *closed* in a run.
    *   **Calculation in `run_experiment.py`:** `evlog[evlog.case_status == "closed"].est_NPS.mean()`

*   **`closed_avg_predicted_throughput_time`**:
    *   **Definition:** The average of `est_throughput_time` values for those cases that were eventually *closed* in a run.
    *   **Calculation in `run_experiment.py`:** `evlog[evlog.case_status == "closed"].est_throughput_time.mean()`

*   **`closed_avg_predicted_NPS_priority`**:
    *   **Definition:** The average of `est_NPS_priority` values for those cases that were eventually *closed* in a run.
    *   **Calculation in `run_experiment.py`:** `evlog[evlog.case_status == "closed"].est_NPS_priority.mean()`

*   **`closed_avg_initial_delay`**:
    *   **Definition:** The average of `initial_delay` values for those cases that were eventually *closed* in a run.
    *   **Calculation in `run_experiment.py`:** `evlog[evlog.case_status == "closed"].initial_delay.mean()`

### B. Metrics Averaged Over **All** Events/Cases (from `evlog`)
These metrics are calculated from the entire `evlog` of a run. For `simulated_NPS` and `simulated_throughput_time`, which are primarily defined at closure, their values might be NaN or 0 for events of non-closed cases. The `.mean()` pandas function typically ignores NaNs. Thus, these `all_avg_` versions might be similar or identical to their `closed_avg_` counterparts if non-closed cases don't contribute valid numbers.

*   **`all_avg_simulated_NPS`**:
    *   **Definition:** Average of `simulated_NPS` from all events in `evlog`.
    *   **Calculation in `run_experiment.py`:** `evlog.simulated_NPS.mean()`

*   **`all_avg_simulated_throughput_time`**:
    *   **Definition:** Average of `simulated_throughput_time` from all events in `evlog`.
    *   **Calculation in `run_experiment.py`:** `evlog.simulated_throughput_time.mean()`

*   **`all_avg_predicted_NPS`**:
    *   **Definition:** Average of `est_NPS` from all events in `evlog`. Predictions may exist even for cases not yet closed.
    *   **Calculation in `run_experiment.py`:** `evlog.est_NPS.mean()`

*   **`all_avg_predicted_throughput_time`**:
    *   **Definition:** Average of `est_throughput_time` from all events in `evlog`.
    *   **Calculation in `run_experiment.py`:** `evlog.est_throughput_time.mean()`

*   **`all_avg_predicted_NPS_priority`**:
    *   **Definition:** Average of `est_NPS_priority` from all events in `evlog`.
    *   **Calculation in `run_experiment.py`:** `evlog.est_NPS_priority.mean()`

*   **`all_avg_initial_delay`**:
    *   **Definition:** Average of `initial_delay` from all events in `evlog`. Initial delay can occur for any case.
    *   **Calculation in `run_experiment.py`:** `evlog.initial_delay.mean()`

### C. Case Count and Other Aggregate Metrics

*   **`cases_arrived`**:
    *   **Definition:** The total number of cases that arrived during the simulation period for a run.
    *   **Calculation in `run_experiment.py`:** `len(Case_DB)`

*   **`cases_closed`**:
    *   **Definition:** The total number of cases that achieved a "closed" status by the end of the simulation run.
    *   **Calculation in `run_experiment.py`:** `len(Case_DB[Case_DB.status == "closed"])`

*   **`case_queued`**:
    *   **Definition:** The total number of unique cases that were put into any queue at least once during the simulation run.
    *   **Calculation in `run_experiment.py`:** `len(Case_DB[Case_DB.case_queued == True])`

*   **`max_tracelen`**:
    *   **Definition:** The maximum number of events (trace length) recorded for any single case within a simulation run.
    *   **Calculation in `run_experiment.py`:** `evlog.groupby('case_id').size().max()` (if `evlog` is not empty)

## III. Derived Report Metrics (Calculated by `report_from_results.py`)

These metrics are calculated within the `analysis/report_from_results.py` script, using the aggregated metrics from `design_table.csv`.

*   **`closed_percent`**:
    *   **Definition:** The percentage of arrived cases that were closed during the simulation run.
    *   **Calculation:** `(cases_closed / cases_arrived) * 100`

*   **`case_queued_percent`**:
    *   **Definition:** The percentage of arrived cases that experienced being queued at some point.
    *   **Calculation:** `(case_queued / cases_arrived) * 100` 