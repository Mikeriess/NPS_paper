#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Queue Prioritization NPS Simulation Analysis Report Generator (using results.csv)

This script generates a PDF report from the consolidated results.csv file,
containing visualizations of NPS scores and queue prioritization methods.

Usage:
    python analysis/report_from_results.py --experiment experiments/test1
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import statsmodels.formula.api as smf # Added import for statsmodels

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10})

def load_results(experiment_dir):
    """Load the design table file and filter for completed runs."""
    results_path = os.path.join(experiment_dir, "design_table.csv") # Changed filename
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Design table file not found at {results_path}") # Updated error message
    df = pd.read_csv(results_path)
    # Filter for completed runs, assuming 'Done' column exists and '1' means completed.
    # If 'Done' column doesn't exist or has different semantics, this part may need adjustment.
    if 'Done' in df.columns:
        df = df[df['Done'] == 1].copy() # Added filtering for completed runs
    return df

def generate_report(experiment_dir, output_pdf=None):
    """
    Generate a PDF report with analysis visualizations.
    """
    # Load the experiment results
    results = load_results(experiment_dir)
    
    # Define a consistent order for priority schemes to use across all plots
    # Using the standard order: FCFS (baseline), LRTF, SRTF, NPS
    priority_order = ['FCFS', 'LRTF', 'SRTF', 'NPS']
    
    # Filter to only include schemes that exist in the data
    available_schemes = sorted(results['F_priority_scheme'].unique())
    ordered_schemes = [scheme for scheme in priority_order if scheme in available_schemes]
    
    # Add any schemes from the data that weren't in our predefined order
    for scheme in available_schemes:
        if scheme not in ordered_schemes:
            ordered_schemes.append(scheme)
    
    # Define output PDF path
    if output_pdf is None:
        output_pdf = os.path.join(experiment_dir, f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    
    # Create PDF
    with PdfPages(output_pdf) as pdf:
        # Create a title page
        plt.figure(figsize=(12, 8))
        plt.axis('off')
        plt.text(0.5, 0.7, 'NPS Queue Prioritization Simulation Analysis', 
                 ha='center', fontsize=24, weight='bold')
        plt.text(0.5, 0.6, f'Experiment: {os.path.basename(os.path.normpath(experiment_dir))}', 
                 ha='center', fontsize=18)
        plt.text(0.5, 0.5, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                 ha='center', fontsize=14)
        pdf.savefig()
        plt.close()
        
        # Summary table
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.text(0.5, 0.97, 'Experiment Summary', ha='center', fontsize=18, weight='bold')
        
        # Create summary statistics
        summary_text = [
            f"Total Runs: {len(results)}", # Will now reflect total *completed* runs if filtering is applied
            f"Priority Schemes: {', '.join(sorted(results['F_priority_scheme'].unique()))}",
        ]
        
        # Add parameter information if available
        if 'F_NPS_dist_bias' in results.columns:
            summary_text.append(f"NPS Bias Values: {', '.join([str(x) for x in sorted(results['F_NPS_dist_bias'].unique())])}")
        
        if 'F_tNPS_wtime_effect_bias' in results.columns:
            summary_text.append(f"Effect Multiplier Values: {', '.join([str(x) for x in sorted(results['F_tNPS_wtime_effect_bias'].unique())])}")
        
        y_pos = 0.92
        for line in summary_text:
            plt.text(0.5, y_pos, line, ha='center', fontsize=14)
            y_pos -= 0.04
        
        # Add more vertical space before "Overall Statistics"
        y_pos -= 0.04
        
        # Add overall statistics
        plt.text(0.5, y_pos, 'Overall Statistics', ha='center', fontsize=16, weight='bold')
        
        # Create a summary table with key metrics - Using all the requested metrics
        # First calculate closed_percent
        # Ensure 'cases_arrived' is not zero to avoid division by zero
        if 'cases_arrived' in results.columns and 'cases_closed' in results.columns:
            results['closed_percent'] = np.where(results['cases_arrived'] > 0, (results['cases_closed'] / results['cases_arrived']) * 100, 0)
        else:
            # Handle cases where these columns might be missing, though they are expected from design_table.csv
            results['closed_percent'] = 0 
        
        overall_stats_cols = [
            # Case metrics
            'cases_arrived', 'cases_assigned_at_end', 'case_queued', 'cases_closed', 'closed_percent',
            # Predicted metrics
            'closed_avg_predicted_NPS', 'closed_avg_predicted_NPS_priority', 
            'closed_avg_predicted_throughput_time',
            # Simulated metrics
            'closed_avg_simulated_NPS', 
            'closed_avg_simulated_throughput_time',
            'closed_avg_initial_delay'
        ]
        # Filter out any columns not present in results to prevent errors during groupby
        available_overall_stats_cols = [col for col in overall_stats_cols if col in results.columns]
        
        overall_stats = results.groupby('F_priority_scheme')[
            available_overall_stats_cols
        ].mean().reset_index()
        
        # Transpose the table - schemes become columns, metrics become rows
        schemes = ordered_schemes
        
        # Add scheme names as column headers with spacing
        x_positions = [0.35, 0.5, 0.65, 0.8]  # Adjust these positions if needed
        for i, scheme in enumerate(schemes):
            plt.text(x_positions[i], y_pos - 0.05, scheme, ha='center', fontsize=12, weight='bold')
        
        # Add row labels (metrics)
        metric_names = [
            'CASE METRICS',
            'Total Cases',            # maps to cases_arrived
            'Assigned Cases',         # maps to cases_assigned_at_end
            'Queued Cases',           # maps to case_queued
            'Closed Cases',           # maps to cases_closed
            'Closed Percent (%)',     # calculated
            
            'PREDICTED METRICS',      # Renamed from ESTIMATED
            'Mean Predicted NPS',     # maps to closed_avg_predicted_NPS
            'Mean Predicted NPS Priority', # maps to closed_avg_predicted_NPS_priority
            'Mean Predicted Throughput Time (h)', # maps to closed_avg_predicted_throughput_time
            
            'SIMULATED METRICS',
            'Mean Simulated NPS',     # maps to closed_avg_simulated_NPS
            'Mean Simulated Throughput Time (h)', # maps to closed_avg_simulated_throughput_time
            'Mean Initial Delay (h)'  # maps to closed_avg_initial_delay
        ]
        
        # Variables to track section headers
        section_headers = ['CASE METRICS', 'PREDICTED METRICS', 'SIMULATED METRICS'] # Updated section header
        
        # Mapping of metrics to their corresponding columns in the dataframe
        metric_map = {
            # Case metrics
            'Total Cases': 'cases_arrived',
            'Assigned Cases': 'cases_assigned_at_end',
            'Queued Cases': 'case_queued',
            'Closed Cases': 'cases_closed',
            'Closed Percent (%)': 'closed_percent',
            
            # Predicted metrics
            'Mean Predicted NPS': 'closed_avg_predicted_NPS',
            'Mean Predicted NPS Priority': 'closed_avg_predicted_NPS_priority',
            'Mean Predicted Throughput Time (h)': 'closed_avg_predicted_throughput_time',
            
            # Simulated metrics
            'Mean Simulated NPS': 'closed_avg_simulated_NPS',
            'Mean Simulated Throughput Time (h)': 'closed_avg_simulated_throughput_time',
            'Mean Initial Delay (h)': 'closed_avg_initial_delay'
        }
        
        # Adjust y positions for the metrics - reduced spacing to fit all metrics
        y_start = y_pos - 0.10
        y_positions = [y_start - 0.025*i for i in range(len(metric_names))]
        
        # Add all metrics to the table
        for i, metric_name in enumerate(metric_names):
            # Make section headers bold and add spacing
            if metric_name in section_headers:
                plt.text(0.01, y_positions[i], metric_name, ha='left', fontsize=12, weight='bold', color='blue')
                continue
                
            # Add regular metric names
            plt.text(0.02, y_positions[i], metric_name, ha='left', fontsize=10)
        
        # Add the values from each scheme in the appropriate cell
        for i, scheme in enumerate(schemes):
            if scheme not in overall_stats['F_priority_scheme'].values:
                continue
                
            scheme_data = overall_stats[overall_stats['F_priority_scheme'] == scheme].iloc[0]
            
            # Add each metric value
            for j, metric_name in enumerate(metric_names):
                # Skip section headers
                if metric_name in section_headers:
                    continue
                
                # Get the corresponding column name from the mapping
                col_name = metric_map[metric_name]
                if col_name in scheme_data:
                    value = scheme_data[col_name]
                    
                    # Format differently based on metric type
                    if 'Percent' in metric_name:
                        formatted_value = f"{value:.1f}%"
                    elif 'NPS' in metric_name or 'Time' in metric_name or 'Delay' in metric_name:
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = f"{value:.0f}"
                    
                    # Find correct position for the value
                    pos_idx = [k for k, name in enumerate(metric_names) if name == metric_name][0]
                    plt.text(x_positions[i], y_positions[pos_idx], formatted_value, ha='center', fontsize=10)
        
        # Increase figure size to fit all the metrics
        plt.gcf().set_size_inches(12, 14)
        
        pdf.savefig()
        plt.close()
        
        # 1. NPS distribution plot
        plot_nps_distribution(results, pdf, ordered_schemes)
        
        # 2 & 3. Performance comparison plots
        plot_performance_comparison(results, pdf, ordered_schemes)
    
    print(f"Report generated: {output_pdf}")
    return output_pdf

def plot_nps_distribution(results, pdf, ordered_schemes):
    """
    Plot NPS distribution by priority scheme.
    """
    plt.figure(figsize=(10, 6))
    
    # Create custom palette for consistent colors
    palette = sns.color_palette("husl", n_colors=len(ordered_schemes))
    scheme_colors = {scheme: palette[i] for i, scheme in enumerate(ordered_schemes)}
    
    # Create a boxplot of NPS scores by priority scheme
    sns.boxplot(x='F_priority_scheme', y='closed_avg_simulated_NPS',  # Updated y argument
                data=results, 
                order=ordered_schemes, 
                palette=scheme_colors,
                showfliers=False) # Hiding outliers for now
    
    plt.title('Distribution of Mean Simulated NPS by Priority Scheme', fontsize=16)
    plt.xlabel('Priority Scheme', fontsize=12)
    plt.ylabel('Mean Simulated NPS (per run)', fontsize=12) # Clarified y-axis label
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Add a table with mean and std dev of NPS scores below the plot
    plt.figure(figsize=(10, 2)) # Adjusted size for table
    plt.axis('off')
    
    if 'closed_avg_simulated_NPS' in results.columns: # Check if column exists
        nps_summary = results.groupby('F_priority_scheme')['closed_avg_simulated_NPS'].agg(['mean', 'std']).reindex(ordered_schemes).reset_index()
        
        table_data = [["Scheme", "Mean of Means", "Std Dev of Means"]]
        for index, row in nps_summary.iterrows():
            table_data.append([row['F_priority_scheme'], f"{row['mean']:.2f}", f"{row['std']:.2f}"])
            
        table = plt.table(cellText=table_data, colLabels=None, cellLoc = 'center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5) # Adjust cell height
        plt.title('Summary of Mean Simulated NPS (across runs)', loc='left', fontsize=10, y=0.8) # Adjusted title
    else:
        plt.text(0.5, 0.5, "closed_avg_simulated_NPS column not found for summary table.", ha='center', va='center')

    plt.tight_layout()
    pdf.savefig()
    plt.close()

def plot_regression_analysis(results_df, pdf_object, dep_var_col, dep_var_label, predictor_cols_numerical, predictor_col_categorical):
    """
    Perform OLS regression and plot the summary to a PDF page.
    """
    plt.figure(figsize=(11.7, 16.5))  # A4 size in inches (landscape for summary)
    plt.axis('off')
    
    formula_parts = [dep_var_col, "~"]
    if predictor_cols_numerical:
        formula_parts.append(" + ".join(predictor_cols_numerical))
    
    if predictor_col_categorical:
        if predictor_cols_numerical: # Add '+' if numerical predictors are already there
            formula_parts.append(f" + C({predictor_col_categorical})")
        else:
            formula_parts.append(f"C({predictor_col_categorical})") # No '+' if it's the first predictor
            
    if len(formula_parts) == 2: # Only dep_var ~ (i.e. no predictors)
        plt.text(0.5, 0.5, f"No valid predictors found for {dep_var_label}.", ha='center', va='center', fontsize=12, color='red')
        plt.title(f"Regression Analysis for: {dep_var_label} - SKIPPED", fontsize=14, y=0.98)
        pdf_object.savefig()
        plt.close()
        return

    formula = "".join(formula_parts)
    
    plt.title(f"Regression Analysis for: {dep_var_label}\\nFormula: {formula}", fontsize=14, y=0.97)

    try:
        # Ensure dependent variable is numeric
        results_df[dep_var_col] = pd.to_numeric(results_df[dep_var_col], errors='coerce')
        # Drop rows where the dependent or any predictor variable is NaN before fitting
        cols_for_na_check = [dep_var_col] + predictor_cols_numerical
        if predictor_col_categorical:
            cols_for_na_check.append(predictor_col_categorical)
        
        cleaned_df = results_df.dropna(subset=cols_for_na_check).copy()

        if cleaned_df.empty or cleaned_df[dep_var_col].nunique() < 2:
            raise ValueError("Not enough data or variance in dependent variable after cleaning.")

        # Check variance for numerical predictors in cleaned_df
        valid_numerical_predictors = []
        for p_col in predictor_cols_numerical:
            if cleaned_df[p_col].nunique() > 1:
                valid_numerical_predictors.append(p_col)
            else:
                print(f"Warning: Predictor '{p_col}' has no variance after NA drop for DV '{dep_var_label}', removing from this regression.")
        
        # Reconstruct formula with only valid predictors
        formula_parts_cleaned = [dep_var_col, "~"]
        if valid_numerical_predictors:
            formula_parts_cleaned.append(" + ".join(valid_numerical_predictors))
        
        if predictor_col_categorical and cleaned_df[predictor_col_categorical].nunique() > 1 :
            if valid_numerical_predictors:
                 formula_parts_cleaned.append(f" + C({predictor_col_categorical})")
            else:
                 formula_parts_cleaned.append(f"C({predictor_col_categorical})")
        elif predictor_col_categorical:
             print(f"Warning: Categorical predictor '{predictor_col_categorical}' has no variance after NA drop for DV '{dep_var_label}', removing from this regression.")


        if len(formula_parts_cleaned) == 2: # Only dep_var ~ (i.e. no predictors left)
             raise ValueError("No valid predictors left after checking variance post NA drop.")
        
        formula_cleaned = "".join(formula_parts_cleaned)


        model = smf.ols(formula_cleaned, data=cleaned_df).fit()
        summary_text = str(model.summary())
        
        # Update title with the formula actually used
        plt.gca().set_title(f"Regression Analysis for: {dep_var_label}\\nFormula: {formula_cleaned}", fontsize=14, y=0.97, wrap=True)
        plt.text(0.01, 0.90, summary_text, family='monospace', fontsize=8, va='top', ha='left')

    except Exception as e:
        error_message = f"Could not perform regression for {dep_var_label}.\\nError: {str(e)}"
        print(error_message)
        plt.text(0.5, 0.5, error_message, ha='center', va='center', fontsize=10, color='red', wrap=True)

    pdf_object.savefig()
    plt.close()

def plot_performance_comparison(results, pdf, ordered_schemes):
    """
    Plot key performance metrics for comparison.
    - Average NPS
    - Average Throughput Time
    - Total Closed Cases
    - Average Initial Delay
    """
    
    # Define metrics to plot, their y-axis labels, and titles
    # Ensure these columns exist in the results DataFrame
    metrics_to_plot_config = {
        'closed_avg_simulated_NPS': ('Average Simulated NPS (per run)', 'Overall Average Simulated NPS by Priority Scheme'),
        'closed_avg_simulated_throughput_time': ('Average Simulated Throughput Time (h, per run)', 'Overall Average Simulated Throughput Time by Priority Scheme'),
        'cases_closed': ('Total Closed Cases (sum across runs)', 'Total Closed Cases by Priority Scheme'), # This will now sum up 'cases_closed' per scheme
        'closed_avg_initial_delay': ('Average Initial Delay (h, per run)', 'Overall Average Initial Delay by Priority Scheme')
    }
    
    # Filter out metrics that are not in results.columns
    available_metrics = {metric: config for metric, config in metrics_to_plot_config.items() if metric in results.columns}

    # Create custom palette for consistent colors
    palette = sns.color_palette("viridis", n_colors=len(ordered_schemes))
    scheme_colors = {scheme: palette[i] for i, scheme in enumerate(ordered_schemes)}

    for metric, (y_label, title) in available_metrics.items():
        plt.figure(figsize=(10, 6))
        
        # For 'cases_closed', we want to sum them up across runs for each scheme. For others, mean.
        if metric == 'cases_closed':
            plot_data = results.groupby('F_priority_scheme')[metric].sum().reindex(ordered_schemes).reset_index()
        else:
            plot_data = results.groupby('F_priority_scheme')[metric].mean().reindex(ordered_schemes).reset_index()

        sns.barplot(x='F_priority_scheme', y=metric, data=plot_data, order=ordered_schemes, palette=scheme_colors, ci=None) # ci=None as we plot means of means or sums
        
        plt.title(title, fontsize=16)
        plt.xlabel('Priority Scheme', fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # --- Regression Analysis Section ---
    if results.empty:
        print("Skipping regression analysis as the results DataFrame is empty.")
    else:
        predictor_cols_numerical = []
        potential_numerical_predictors = ['F_NPS_dist_bias', 'F_tNPS_wtime_effect_bias', 'F_number_of_agents'] # Add other potential factors if needed
        
        for col_name in potential_numerical_predictors:
            if col_name in results.columns and pd.to_numeric(results[col_name], errors='coerce').nunique() > 1:
                predictor_cols_numerical.append(col_name)
            elif col_name in results.columns:
                 print(f"Predictor '{col_name}' excluded from regression due to single unique value or non-numeric type.")


        categorical_predictor = 'F_priority_scheme' if 'F_priority_scheme' in results.columns and results['F_priority_scheme'].nunique() > 1 else None
        if 'F_priority_scheme' in results.columns and results['F_priority_scheme'].nunique() <=1 and categorical_predictor is None:
            print("Categorical predictor 'F_priority_scheme' excluded due to single unique value.")


        dependent_vars_config = {
            'closed_avg_simulated_throughput_time': 'Simulated Throughput Time',
            'closed_percent': 'Proportion of Closed Cases (%)',
            'closed_avg_simulated_NPS': 'Simulated NPS'
        }

        min_obs_for_regression = len(predictor_cols_numerical) + (results[categorical_predictor].nunique() if categorical_predictor and categorical_predictor in results else 1) + 2 # Basic rule of thumb: k+1 obs, adding a small margin

        for dep_var_col, dep_var_label in dependent_vars_config.items():
            if dep_var_col in results.columns and pd.to_numeric(results[dep_var_col], errors='coerce').notnull().all():
                # Ensure the dependent variable itself has variance
                if results[dep_var_col].nunique() < 2:
                    print(f"Skipping regression for '{dep_var_label}' (column '{dep_var_col}') as it has less than 2 unique values.")
                    continue
                
                # Check if there are enough data points
                if len(results.dropna(subset=[dep_var_col] + predictor_cols_numerical + ([categorical_predictor] if categorical_predictor else []))) >= min_obs_for_regression:
                    plot_regression_analysis(results.copy(), pdf, dep_var_col, dep_var_label, predictor_cols_numerical, categorical_predictor)
                else:
                    print(f"Skipping regression for '{dep_var_label}' due to insufficient data points after potential NA drops (need at least {min_obs_for_regression}).")
            else:
                print(f"Skipping regression for '{dep_var_label}' as column '{dep_var_col}' is missing, not numeric, or all NaN.")
        
        # --- End of Regression Analysis Section ---

def main():
    """Main function to parse arguments and generate the report."""
    parser = argparse.ArgumentParser(description='Generate analysis report from results.csv for NPS simulation experiments.')
    parser.add_argument('--experiment', required=True, help='Path to the experiment directory')
    parser.add_argument('--output', help='Output PDF file path (optional)')
    
    args = parser.parse_args()
    
    generate_report(args.experiment, args.output)

if __name__ == "__main__":
    main() 