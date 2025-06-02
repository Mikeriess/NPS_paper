#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate experiment design table from a JSON configuration file.
"""

import json
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from helpers.DoE import build_full_fact, fix_label_values

def create_example_json():
    """Create an example JSON file with default experiment settings."""
    example_settings = {
        "F_priority_scheme": ["NPS", "SRTF", "LRTF", "FCFS"],
        "F_number_of_agents": [3],
        "F_hard_ceiling": ["NONE"],
        "F_ceiling_value": [2.5],
        "F_burn_in": [0],
        "F_days": [50],
        "F_fit_on_burn_in": ["Static", "Train"],
        "F_NPS_dist_bias": [-2, -1, -0.5, 0, 0.5, 1, 2],
        "F_tNPS_wtime_effect_bias": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        "startdate": ["2018-07-01"],
        "repetition": [0]
    }
    
    with open("example.json", "w") as f:
        json.dump(example_settings, f, indent=4)
    
    print(f"Created example.json in {os.getcwd()}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate experiment design table from JSON configuration.")
    parser.add_argument("--settings", type=str, help="Path to the JSON settings file")
    args = parser.parse_args()
    
    if not args.settings:
        print("Error: No settings file provided.")
        print("Usage: python generate_design.py --settings experiments/test1/settings.json")
        create_example_json()
        sys.exit(1)
    
    # Check if settings file exists
    if not os.path.exists(args.settings):
        print(f"Error: Settings file '{args.settings}' not found.")
        create_example_json()
        sys.exit(1)
    
    # Get the directory where the settings file is located
    settings_path = Path(args.settings)
    output_dir = settings_path.parent
    
    # Load settings from JSON
    try:
        with open(args.settings, "r") as f:
            run_settings = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{args.settings}'.")
        create_example_json()
        sys.exit(1)
    
    # Generate a full factorial design
    df = build_full_fact(run_settings)
    
    # Get string values back
    df = fix_label_values(df, run_settings, variables=["F_priority_scheme", "F_hard_ceiling", "F_fit_on_burn_in", "startdate"])
    
    # If burn in period is incorrectly specified, increase days by burn-in period
    for day in df.index:
        if df.loc[day, "F_burn_in"] <= df.loc[day, "F_days"]:
            df.loc[day, "F_days"] = df.loc[day, "F_days"] + df.loc[day, "F_burn_in"]
    
    # Change dtypes
    df.F_number_of_agents = df.F_number_of_agents.astype(int)
    df.repetition = df.repetition.astype(int)
    df.F_days = df.F_days.astype(int)
    df.F_burn_in = df.F_burn_in.astype(int)
    # F_fit_on_burn_in is now handled as string, no conversion needed
    
    # Placeholder variables
    float_placeholder_variables = [
        'closed_avg_simulated_NPS', 'closed_avg_simulated_throughput_time',
        'closed_avg_predicted_NPS', 'closed_avg_predicted_throughput_time',
        'closed_avg_predicted_NPS_priority', 'closed_avg_initial_delay',
        'closed_avg_activity_start_delay', 'closed_avg_duration_delayed',
        'all_avg_simulated_NPS', 'all_avg_simulated_throughput_time',
        'all_avg_predicted_NPS', 'all_avg_predicted_throughput_time',
        'all_avg_predicted_NPS_priority', 'all_avg_initial_delay',
        'all_avg_activity_start_delay', 'all_avg_duration_delayed',
        'Simulation_duration_min',
        'dynamic_model_mae_burnin', 'dynamic_model_mse_burnin',
        'dynamic_model_mae_main', 'dynamic_model_mse_main'
    ]
    
    integer_placeholder_variables = [
        'cases_arrived', 'cases_closed', 'case_queued',
        'cases_assigned_at_end', 'min_tracelen', 'max_tracelen',
        'dynamic_model_n_burnin_samples', 'dynamic_model_n_main_cases'
    ]

    for var in float_placeholder_variables:
        df[var] = np.nan

    for var in integer_placeholder_variables:
        df[var] = -1
    
    # Sort experiments and add metadata
    df["RUN"] = list(range(0, len(df)))
    df["Done"] = 0
    
    print(df)
    
    # Save design table to the same directory as the settings file
    design_table_path = output_dir / "design_table.csv"
    df.to_csv(design_table_path, index=False)
    print(f"Design table saved to {design_table_path}")

if __name__ == "__main__":
    main() 