# Report Metric Definitions

This page explains the key metrics presented in the subsequent plots and tables of this report.

*   **`closed_avg_simulated_NPS`**: Average Net Promoter Score for closed cases in a simulation run. Calculated based on the simulated throughput time and case topic, then averaged. (Range: -100 to 100, higher is better).
*   **`closed_avg_simulated_throughput_time`**: Average time (in simulation hours) from case creation to case closure for all cases that were closed within a simulation run. (Lower is generally better, indicating faster resolution).
*   **`closed_percent`**: The percentage of cases that arrived during a simulation run and were subsequently closed within that same run. Calculated as `(cases_closed / cases_arrived) * 100`. (Range: 0-100%, higher indicates better system throughput for arriving cases).
*   **`case_queued_percent`**: The percentage of cases that arrived during a simulation run and had to enter a queue at least once before being processed. Calculated as `(case_queued / cases_arrived) * 100`. (Range: 0-100%, lower indicates less waiting and potentially higher immediate service levels).
*   **`closed_avg_initial_delay`**: The average initial waiting time (in simulation hours) experienced by cases that were eventually closed within a simulation run. This measures the time from case arrival until the first agent activity begins on that case. (Lower is better).
*   **`closed_avg_predicted_throughput_time`**: At the moment of case arrival, this is the average *predicted* throughput time for those cases that were eventually closed in a simulation run. This reflects the system's initial estimation.
*   **`closed_avg_predicted_NPS`**: At the moment of case arrival, this is the average *predicted* Net Promoter Score for those cases that were eventually closed in a simulation run. This is based on the initial prediction of throughput time.
*   **`closed_avg_predicted_NPS_priority`**: At the moment of case arrival, this is the average *predicted* NPS-based priority score for cases that were eventually closed in a simulation run. This score might be used by NPS-based prioritization schemes.
*   **`max_tracelen`**: The maximum number of events recorded for any single case within a simulation run. This can give an indication of the most complex or lengthy case interaction in a run.
*   **`all_avg_initial_delay`**: The average initial waiting time (in simulation hours) for *all* cases that arrived during a simulation run, regardless of whether they were closed or not by the end of the run. (Lower is better).
*   **`cases_closed`**: The total count of cases that were successfully closed within a simulation run. (Higher generally indicates more system output). 