#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Aggregated Results Table

This script processes experiment run data to create an aggregated results table.
It calculates statistics from individual run log files and case databases, then
combines these with the design table factors to create a comprehensive results.csv file.

Usage:
    python analysis/generate_results.py --experiment experiments/test1
"""

import os
import argparse
import pandas as pd
import numpy as np
import glob
from datetime import datetime

def load_design_table(experiment_dir):
    """Load the experiment design table."""
    design_path = os.path.join(experiment_dir, "design_table.csv")
    if not os.path.exists(design_path):
        raise FileNotFoundError(f"Design table not found at {design_path}")
    return pd.read_csv(design_path)

def process_log_file(log_path):
    """Process a log file to extract metrics."""
    if not os.path.exists(log_path):
        print(f"Warning: Log file not found at {log_path}")
        return None

    try:
        log_df = pd.read_csv(log_path)
    except Exception as e:
        print(f"Error reading log file {log_path}: {e}")
        return None

    metrics = {}
    # Columns to calculate statistics for
    columns = [
        'simulated_NPS', 
        'simulated_NPS_priority', 
        'simulated_throughput_time', 
        'initial_delay', 
        'est_NPS', 
        'est_NPS_priority'
    ]
    
    for col in columns:
        if col in log_df.columns:
            metrics[f'mean_{col}'] = log_df[col].mean()
            metrics[f'stdev_{col}'] = log_df[col].std()
            metrics[f'min_{col}'] = log_df[col].min()
            metrics[f'max_{col}'] = log_df[col].max()
        else:
            print(f"Warning: Column {col} not found in log file {log_path}")
            metrics[f'mean_{col}'] = np.nan
            metrics[f'stdev_{col}'] = np.nan
            metrics[f'min_{col}'] = np.nan
            metrics[f'max_{col}'] = np.nan
    
    return metrics

def process_case_db(case_db_path):
    """Process a case database file to extract metrics."""
    if not os.path.exists(case_db_path):
        print(f"Warning: Case DB file not found at {case_db_path}")
        return None

    try:
        case_df = pd.read_csv(case_db_path)
    except Exception as e:
        print(f"Error reading case DB file {case_db_path}: {e}")
        return None

    metrics = {}
    
    # Calculate statistics for est_throughputtime
    if 'est_throughputtime' in case_df.columns:
        metrics['mean_est_throughputtime'] = case_df['est_throughputtime'].mean()
        metrics['stdev_est_throughputtime'] = case_df['est_throughputtime'].std()
        metrics['min_est_throughputtime'] = case_df['est_throughputtime'].min()
        metrics['max_est_throughputtime'] = case_df['est_throughputtime'].max()
    else:
        print(f"Warning: Column est_throughputtime not found in case DB {case_db_path}")
        metrics['mean_est_throughputtime'] = np.nan
        metrics['stdev_est_throughputtime'] = np.nan
        metrics['min_est_throughputtime'] = np.nan
        metrics['max_est_throughputtime'] = np.nan
    
    # Count metrics
    metrics['Total_cases'] = len(case_df)
    
    if 'case_currently_assigned' in case_df.columns:
        metrics['assigned_cases'] = case_df['case_currently_assigned'].sum()
    else:
        print(f"Warning: Column case_currently_assigned not found in case DB {case_db_path}")
        metrics['assigned_cases'] = np.nan
    
    if 'case_queued' in case_df.columns:
        metrics['queued_cases'] = case_df['case_queued'].sum()
    else:
        print(f"Warning: Column case_queued not found in case DB {case_db_path}")
        metrics['queued_cases'] = np.nan
    
    if 'status' in case_df.columns:
        metrics['closed_cases'] = (case_df['status'] == 'closed').sum()
    else:
        print(f"Warning: Column status not found in case DB {case_db_path}")
        metrics['closed_cases'] = np.nan
    
    return metrics

def generate_results(experiment_dir):
    """Generate the aggregated results table."""
    # Load design table
    design_df = load_design_table(experiment_dir)
    
    # Initialize results list
    results = []
    
    # Process each run
    for _, row in design_df.iterrows():
        run_id = row['RUN']
        run_dir = os.path.join(experiment_dir, str(run_id))
        
        # Skip if run directory doesn't exist
        if not os.path.exists(run_dir):
            print(f"Warning: Run directory not found for RUN={run_id}")
            continue
        
        # Extract run design factors
        run_result = {col: row[col] for col in design_df.columns if col.startswith('F_') or col == 'RUN'}
        
        # Process log file
        log_path = os.path.join(run_dir, f"{run_id}_log.csv")
        log_metrics = process_log_file(log_path)
        if log_metrics:
            run_result.update(log_metrics)
        
        # Process case DB
        case_db_path = os.path.join(run_dir, f"{run_id}_case_DB.csv")
        case_metrics = process_case_db(case_db_path)
        if case_metrics:
            run_result.update(case_metrics)
        
        results.append(run_result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = os.path.join(experiment_dir, "results.csv")
    results_df.to_csv(output_path, index=False)
    
    print(f"Generated results table with {len(results_df)} entries at {output_path}")
    return results_df

def main():
    """Main function to parse arguments and generate results."""
    parser = argparse.ArgumentParser(description='Generate aggregated results table for NPS simulation experiments.')
    parser.add_argument('--experiment', required=True, help='Path to the experiment directory')
    
    args = parser.parse_args()
    
    generate_results(args.experiment)

if __name__ == "__main__":
    main() 