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

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10})

def load_results(experiment_dir):
    """Load the results file."""
    results_path = os.path.join(experiment_dir, "results.csv")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found at {results_path}")
    return pd.read_csv(results_path)

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
            f"Total Runs: {len(results)}",
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
        results['closed_percent'] = results['closed_cases'] / results['Total_cases'] * 100
        
        overall_stats = results.groupby('F_priority_scheme')[
            # Case metrics
            ['Total_cases', 'assigned_cases', 'queued_cases', 'closed_cases', 'closed_percent',
             # Estimated metrics
             'mean_est_NPS', 'stdev_est_NPS', 'mean_est_NPS_priority', 
             'mean_est_throughputtime', 'stdev_est_throughputtime',
             # Simulated metrics
             'mean_simulated_NPS', 'stdev_simulated_NPS', 'mean_simulated_NPS_priority',
             'mean_simulated_throughput_time', 'stdev_simulated_throughput_time',
             'mean_initial_delay', 'stdev_initial_delay']
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
            'Total Cases',
            'Assigned Cases',
            'Queued Cases',
            'Closed Cases',
            'Closed Percent (%)',
            
            'ESTIMATED METRICS',
            'Mean Estimated NPS',
            'Std Dev Estimated NPS',
            'Mean Estimated NPS Priority',
            'Mean Estimated Throughput Time (h)',
            'Std Dev Estimated Throughput Time (h)',
            
            'SIMULATED METRICS',
            'Mean Simulated NPS',
            'Std Dev Simulated NPS',
            'Mean Simulated NPS Priority',
            'Mean Simulated Throughput Time (h)',
            'Std Dev Simulated Throughput Time (h)',
            'Mean Initial Delay (h)',
            'Std Dev Initial Delay (h)'
        ]
        
        # Variables to track section headers
        section_headers = ['CASE METRICS', 'ESTIMATED METRICS', 'SIMULATED METRICS']
        
        # Mapping of metrics to their corresponding columns in the dataframe
        metric_map = {
            # Case metrics
            'Total Cases': 'Total_cases',
            'Assigned Cases': 'assigned_cases',
            'Queued Cases': 'queued_cases',
            'Closed Cases': 'closed_cases',
            'Closed Percent (%)': 'closed_percent',
            
            # Estimated metrics
            'Mean Estimated NPS': 'mean_est_NPS',
            'Std Dev Estimated NPS': 'stdev_est_NPS',
            'Mean Estimated NPS Priority': 'mean_est_NPS_priority',
            'Mean Estimated Throughput Time (h)': 'mean_est_throughputtime',
            'Std Dev Estimated Throughput Time (h)': 'stdev_est_throughputtime',
            
            # Simulated metrics
            'Mean Simulated NPS': 'mean_simulated_NPS',
            'Std Dev Simulated NPS': 'stdev_simulated_NPS',
            'Mean Simulated NPS Priority': 'mean_simulated_NPS_priority',
            'Mean Simulated Throughput Time (h)': 'mean_simulated_throughput_time',
            'Std Dev Simulated Throughput Time (h)': 'stdev_simulated_throughput_time',
            'Mean Initial Delay (h)': 'mean_initial_delay',
            'Std Dev Initial Delay (h)': 'stdev_initial_delay'
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
    sns.boxplot(x='F_priority_scheme', y='mean_simulated_NPS', 
               data=results, order=ordered_schemes, palette=scheme_colors)
    
    plt.title('Distribution of Mean NPS Scores by Priority Scheme', fontsize=14)
    plt.xlabel('F_priority_scheme', fontsize=12)
    plt.ylabel('mean_simulated_NPS', fontsize=12)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # If we have multiple parameter combinations, show heatmap
    if ('F_NPS_dist_bias' in results.columns and 
        'F_tNPS_wtime_effect_bias' in results.columns and
        len(results['F_NPS_dist_bias'].unique()) > 1 and
        len(results['F_tNPS_wtime_effect_bias'].unique()) > 1):
        
        plt.figure(figsize=(12, 10))
        
        # Create a facet grid for different priority schemes
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        
        for i, scheme in enumerate(ordered_schemes):
            if i >= 4:  # Only show first 4 schemes
                break
                
            # Get data for this scheme
            scheme_data = results[results['F_priority_scheme'] == scheme]
            
            # Create pivot table
            if len(scheme_data) > 0:
                pivot = scheme_data.pivot_table(
                    index='F_NPS_dist_bias',
                    columns='F_tNPS_wtime_effect_bias',
                    values='mean_simulated_NPS',
                    aggfunc='mean'
                )
                
                # Plot heatmap
                sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[i])
                axes[i].set_title(f'{scheme}')
                axes[i].set_xlabel('F_tNPS_wtime_effect_bias')
                axes[i].set_ylabel('F_NPS_dist_bias')
        
        plt.tight_layout()
        plt.suptitle('NPS Score by Parameter Combination', fontsize=16, y=1.02)
        pdf.savefig()
        plt.close()

def plot_performance_comparison(results, pdf, ordered_schemes):
    """
    Plot basic performance metrics across priority schemes.
    """
    # Enhanced comparison: NPS metrics side by side
    plt.figure(figsize=(12, 8))
    
    # Create a dataframe with both simulated and estimated NPS metrics
    nps_data = results.groupby('F_priority_scheme')[
        ['mean_simulated_NPS', 'mean_est_NPS', 
         'mean_simulated_NPS_priority', 'mean_est_NPS_priority']
    ].mean().reset_index()
    
    # Reshape for easier plotting
    nps_plot_data = pd.melt(
        nps_data,
        id_vars=['F_priority_scheme'],
        value_vars=['mean_simulated_NPS', 'mean_est_NPS', 
                   'mean_simulated_NPS_priority', 'mean_est_NPS_priority'],
        var_name='Metric',
        value_name='NPS Score'
    )
    
    # Create better labels for the metrics
    metric_labels = {
        'mean_simulated_NPS': 'Simulated NPS',
        'mean_est_NPS': 'Estimated NPS',
        'mean_simulated_NPS_priority': 'Simulated NPS Priority',
        'mean_est_NPS_priority': 'Estimated NPS Priority'
    }
    nps_plot_data['Metric'] = nps_plot_data['Metric'].map(metric_labels)
    
    # Plot NPS comparison
    sns.barplot(x='F_priority_scheme', y='NPS Score', hue='Metric', 
                data=nps_plot_data, order=ordered_schemes)
    
    plt.title('NPS Metrics by Priority Scheme', fontsize=14)
    plt.xlabel('Priority Scheme', fontsize=12)
    plt.ylabel('NPS Score', fontsize=12)
    plt.legend(title='Metric Type')
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Plot throughput time and delay metrics
    plt.figure(figsize=(12, 8))
    time_data = results.groupby('F_priority_scheme')[
        ['mean_simulated_throughput_time', 'mean_est_throughputtime', 'mean_initial_delay']
    ].mean().reset_index()
    
    # Reshape for plotting
    time_plot_data = pd.melt(
        time_data,
        id_vars=['F_priority_scheme'],
        value_vars=['mean_simulated_throughput_time', 'mean_est_throughputtime', 'mean_initial_delay'],
        var_name='Metric',
        value_name='Time (hours)'
    )
    
    # Create better labels
    time_labels = {
        'mean_simulated_throughput_time': 'Simulated Throughput Time',
        'mean_est_throughputtime': 'Estimated Throughput Time',
        'mean_initial_delay': 'Initial Delay'
    }
    time_plot_data['Metric'] = time_plot_data['Metric'].map(time_labels)
    
    # Plot time comparison
    sns.barplot(x='F_priority_scheme', y='Time (hours)', hue='Metric', 
                data=time_plot_data, order=ordered_schemes)
    
    plt.title('Time Metrics by Priority Scheme', fontsize=14)
    plt.xlabel('Priority Scheme', fontsize=12)
    plt.ylabel('Time (hours)', fontsize=12)
    plt.legend(title='Metric Type')
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Plot case metrics with percentage
    plt.figure(figsize=(12, 8))
    
    # Create a bar chart for case counts
    case_data = results.groupby('F_priority_scheme')[
        ['Total_cases', 'closed_cases', 'queued_cases', 'assigned_cases']
    ].mean().reset_index()
    
    melted_case_data = pd.melt(
        case_data,
        id_vars=['F_priority_scheme'],
        value_vars=['Total_cases', 'closed_cases', 'queued_cases', 'assigned_cases'],
        var_name='Metric',
        value_name='Count'
    )
    
    # Better labels
    case_labels = {
        'Total_cases': 'Total Cases',
        'closed_cases': 'Closed Cases',
        'queued_cases': 'Queued Cases',
        'assigned_cases': 'Assigned Cases'
    }
    melted_case_data['Metric'] = melted_case_data['Metric'].map(case_labels)
    
    # Plot case counts
    sns.barplot(x='F_priority_scheme', y='Count', hue='Metric',
                data=melted_case_data, order=ordered_schemes)
    
    plt.title('Case Metrics by Priority Scheme', fontsize=14)
    plt.xlabel('Priority Scheme', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Metric')
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Create a separate plot for closed percentage
    plt.figure(figsize=(10, 6))
    closed_pct_data = results.groupby('F_priority_scheme')['closed_percent'].mean().reset_index()
    
    sns.barplot(x='F_priority_scheme', y='closed_percent', 
                data=closed_pct_data, order=ordered_schemes)
    
    plt.title('Case Closure Rate by Priority Scheme', fontsize=14)
    plt.xlabel('Priority Scheme', fontsize=12)
    plt.ylabel('Closed Cases (%)', fontsize=12)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
    # Add a side-by-side variance comparison (standard deviations)
    plt.figure(figsize=(12, 8))
    
    # Get standard deviation metrics
    std_data = results.groupby('F_priority_scheme')[
        ['stdev_simulated_NPS', 'stdev_est_NPS', 
         'stdev_simulated_throughput_time', 'stdev_est_throughputtime',
         'stdev_initial_delay']
    ].mean().reset_index()
    
    # Reshape for plotting
    std_plot_data = pd.melt(
        std_data,
        id_vars=['F_priority_scheme'],
        value_vars=['stdev_simulated_NPS', 'stdev_est_NPS', 
                   'stdev_simulated_throughput_time', 'stdev_est_throughputtime',
                   'stdev_initial_delay'],
        var_name='Metric',
        value_name='Standard Deviation'
    )
    
    # Better labels
    std_labels = {
        'stdev_simulated_NPS': 'Simulated NPS',
        'stdev_est_NPS': 'Estimated NPS',
        'stdev_simulated_throughput_time': 'Simulated Throughput Time',
        'stdev_est_throughputtime': 'Estimated Throughput Time',
        'stdev_initial_delay': 'Initial Delay'
    }
    std_plot_data['Metric'] = std_plot_data['Metric'].map(std_labels)
    
    # Plot standard deviations
    sns.barplot(x='F_priority_scheme', y='Standard Deviation', hue='Metric',
                data=std_plot_data, order=ordered_schemes)
    
    plt.title('Metric Variability by Priority Scheme', fontsize=14)
    plt.xlabel('Priority Scheme', fontsize=12)
    plt.ylabel('Standard Deviation', fontsize=12)
    plt.legend(title='Metric')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

def main():
    """Main function to parse arguments and generate the report."""
    parser = argparse.ArgumentParser(description='Generate analysis report from results.csv for NPS simulation experiments.')
    parser.add_argument('--experiment', required=True, help='Path to the experiment directory')
    parser.add_argument('--output', help='Output PDF file path (optional)')
    
    args = parser.parse_args()
    
    generate_report(args.experiment, args.output)

if __name__ == "__main__":
    main() 