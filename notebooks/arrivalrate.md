## Inter-arrival Time Model Summary

The simulation calculates the time until the next case arrival using an **exponential regression model**.

1.  **Model Type**: Predicts inter-arrival time, which follows an exponential distribution. The *rate* of this distribution (and thus the expected inter-arrival time) varies.
2.  **Prediction Equation**: 

    `inter_arrival_time_hours = random_exponential_draw * exp(intercept + B_year*year + B_month*month + B_day*day + B_weekday*weekday)`
    
    The `random_exponential_draw` comes from `-log(1-U(0,1))`.
3.  **Predictors**: `year`, `month`, `day` (of month), `weekday`.
4.  **Coefficients**: Hardcoded intercept and beta values for each predictor.
5.  **Output**: Inter-arrival time in hours, subsequently converted to days.

This allows arrival intensity to change based on the date, driven by the regression model and its coefficients. All arrivals are pre-generated for the entire simulation period.
