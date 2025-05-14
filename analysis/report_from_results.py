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
import sys # Import sys to modify path

# --- Add project root to sys.path to allow absolute imports from analysis package ---
# This assumes report_from_results.py is in a subdirectory (e.g., 'analysis') of the project root.
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir)
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
# ---- Simpler way if script is always one level down ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End of sys.path modification ---

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # Added for reading image
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import statsmodels.formula.api as smf # Added import for statsmodels
import re # Added import for regex
import textwrap # Added import for text wrapping
import plotly.graph_objects as go # Added for Plotly table
import json # Added for loading settings.json
from analysis.prediction_error_utils import calculate_and_add_prediction_errors # Reverted to absolute import

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10})

# --- Functions to add Metric Definitions Page --- START
def parse_metric_definitions(filepath):
    """
    Parse metric definitions from a structured markdown file.
    Expects lines like: *   **`metric_key`**: Description
    """
    definitions = {}
    try:
        with open(filepath, 'r') as f:
            content = f.readlines()
        
        # Regex to capture the metric key and its description
        # It looks for: optional spaces, asterisk, spaces, two asterisks, backtick, key, backtick, two asterisks, colon, space, description
        pattern = re.compile(r"^\s*\*\s*\*\*`([^`]+)`\*\*:\s*(.+)$")
        
        for line in content:
            match = pattern.match(line.strip())
            if match:
                key = match.group(1).strip()
                description = match.group(2).strip()
                definitions[key] = description
    except FileNotFoundError:
        print(f"Error: Metric definitions file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error parsing metric definitions file {filepath}: {e}")
        return None
    return definitions

def add_metric_definitions_page(pdf, metric_definitions):
    """
    Add a page to the PDF with metric definitions.
    """
    # Create a new figure for this specific page with a specified DPI
    fig = plt.figure(figsize=(11.7, 8.27), dpi=300) # A4 landscape, explicit DPI
    plt.axis('off') # Turn off axis for the entire figure page

    page_title = "Metric Definitions"
    # Intro text will be part of the Plotly table's title or not included if table is self-explanatory
    
    # Convert definitions to DataFrame
    df_definitions = pd.DataFrame(list(metric_definitions.items()), columns=['Metric', 'Description'])

    # Pre-process descriptions for HTML line breaks for Plotly table cell wrapping
    # Estimate a reasonable wrap width for the description column
    # This is a bit of trial and error based on typical A4 page proportions
    # and font sizes used by Plotly. Let's aim for about 100-120 chars.
    description_wrap_width = 100 # Reduced from 110
    df_definitions['Metric'] = df_definitions['Metric'].apply(
        lambda x: '<br>'.join(textwrap.wrap(x, width=30)) # Wrap metric names too if they get very long
    )
    df_definitions['Description'] = df_definitions['Description'].apply(
        lambda x: '<br>'.join(textwrap.wrap(x, width=description_wrap_width))
    )

    # Create Plotly table figure
    table_fig = go.Figure(data=[go.Table(
        columnwidth = [30,70], 
        header=dict(values=['<b>Metric</b>', '<b>Description</b>'],
                    fill_color='paleturquoise',
                    align=['left', 'left'], font=dict(size=14)), # Increased header font size to 14
        cells=dict(values=[df_definitions.Metric, df_definitions.Description],
                   fill_color='lavender',
                   align=['left', 'left'],
                   font=dict(size=12), # Increased cell font size to 12
                   height=55  # Increased cell height to 55
                  )
    )])

    table_fig.update_layout(
        title_text=page_title,
        title_x=0.5,
        margin=dict(l=20, r=20, t=60, b=20), # Increased top margin for title
        # Explicitly set a height and width for the table image to guide aspect ratio
        width=1100, # Adjusted width for better A4 landscape fit (approx 11.7in * 96 DPI)
        height=770  # Adjusted height for better A4 landscape fit (approx 8.27in * 96 DPI)
    )

    temp_image_path = "temp_metrics_table.png"
    try:
        # Using scale=6 for very high-resolution source PNG
        table_fig.write_image(temp_image_path, scale=6, engine='kaleido') 
        
        img = mpimg.imread(temp_image_path)
        
        # Display the image. imshow will scale it to fit the axes 
        # (which cover the figure since axis is off).
        plt.imshow(img)
        # No need for fig.clf() as it's a new figure for this page
        # No need for plt.axis('off') again as it's done for the figure
        
        pdf.savefig(fig) # Save the Matplotlib figure (which now contains the table image)
    except Exception as e:
        print(f"Error generating or adding metric definitions table image: {e}")
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path) # Clean up temporary file

    # No plt.close(fig) here if fig is the one passed from generate_report for this page.
    # If we created a new fig = plt.figure() specific for this page, then we should close it.
    # For now, assuming fig is the current page canvas. Let's be explicit.
    # The original `add_metric_definitions_page` created its own `fig`. We should stick to that.
    # Corrected logic from thought process: this function *should* create and close its own fig.
    # The `fig` variable used above `fig = plt.figure(figsize=(11.7, 8.27))` was local.
    # The pdf.savefig(fig) call uses this local fig.
    # So, plt.close(fig) is correct if fig is defined locally as it was.
    plt.close(fig) # Close the local figure used for this page.

# --- Functions to add Metric Definitions Page --- END

# --- Functions to add Experiment Settings Page --- START
def load_experiment_settings(experiment_dir):
    """
    Load experiment settings from settings.json in the experiment directory.
    """
    settings_path = os.path.join(experiment_dir, "settings.json")
    settings_list = []
    try:
        with open(settings_path, 'r') as f:
            data = json.load(f)
        for key, value in data.items():
            if isinstance(value, list):
                levels_str = ', '.join(map(str, value)) # Convert list of levels to string
            else:
                levels_str = str(value)
            settings_list.append([key, levels_str])
    except FileNotFoundError:
        print(f"Error: settings.json not found at {settings_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {settings_path}")
        return None
    except Exception as e:
        print(f"Error loading or parsing settings.json: {e}")
        return None
    return settings_list

def add_settings_table_page(pdf, settings_list, experiment_name):
    """
    Add a page to the PDF with experiment settings table.
    """
    fig = plt.figure(figsize=(11.7, 8.27), dpi=300) # A4 landscape, explicit DPI
    plt.axis('off')

    page_title = f"Experiment Design Factors & Levels: {experiment_name}"
    
    df_settings = pd.DataFrame(settings_list, columns=['Factor', 'Levels'])

    factor_wrap_width = 35
    levels_wrap_width = 90 

    df_settings['Factor'] = df_settings['Factor'].apply(
        lambda x: '<br>'.join(textwrap.wrap(x, width=factor_wrap_width))
    )
    df_settings['Levels'] = df_settings['Levels'].apply(
        lambda x: '<br>'.join(textwrap.wrap(x, width=levels_wrap_width))
    )

    table_fig = go.Figure(data=[go.Table(
        columnwidth = [30,70], # Factor: 30%, Levels: 70%
        header=dict(values=['<b>Factor</b>', '<b>Levels</b>'],
                    fill_color='lightskyblue', # Different color for this table header
                    align=['left', 'left'], font=dict(size=14)), # Increased header font size to 14
        cells=dict(values=[df_settings.Factor, df_settings.Levels],
                   fill_color='lightcyan',   # Different color for these cells
                   align=['left', 'left'],
                   font=dict(size=12), # Increased cell font size to 12
                   height=55 # Slightly increased cell height for larger font
                  )
    )])

    table_fig.update_layout(
        title_text=page_title,
        title_x=0.5,
        margin=dict(l=20, r=20, t=60, b=20),
        width=1100, 
        height=770  
    )

    temp_image_path = "temp_settings_table.png"
    try:
        table_fig.write_image(temp_image_path, scale=6, engine='kaleido') 
        
        img = mpimg.imread(temp_image_path)
        plt.imshow(img)
        pdf.savefig(fig)
    except Exception as e:
        print(f"Error generating or adding settings table image: {e}")
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path) 
    plt.close(fig) 
# --- Functions to add Experiment Settings Page --- END

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
    print("DEBUG: ===== ENTERING generate_report FUNCTION =====")
    # Load the experiment results
    results = load_results(experiment_dir)
    
    if results.empty:
        print(f"No completed runs (Done == 1) found in {os.path.join(experiment_dir, 'design_table.csv')}.")
        print("Report cannot be generated. Please complete some experiment runs first.")
        return None
    
    # Calculate additional metrics if needed
    if 'cases_arrived' in results.columns and 'cases_closed' in results.columns:
        results['closed_percent'] = np.where(results['cases_arrived'] > 0, 
                                           (results['cases_closed'] / results['cases_arrived']) * 100, 0)
    if 'cases_arrived' in results.columns and 'case_queued' in results.columns:
        results['case_queued_percent'] = np.where(results['cases_arrived'] > 0,
                                                (results['case_queued'] / results['cases_arrived']) * 100, 0)
    
    # Calculate and add prediction error metrics
    print("INFO: Calculating prediction error metrics...")
    results = calculate_and_add_prediction_errors(results, experiment_dir)
    print("INFO: Prediction error metrics calculation complete.")

    # Save the processed results as experiment.csv
    experiment_csv_path = os.path.join(experiment_dir, "experiment.csv")
    results.to_csv(experiment_csv_path, index=False)
    print(f"Experiment data saved to: {experiment_csv_path}")
    
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
        
        # Add Metric Definitions Page
        metric_definitions_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_metric_definitions.md")
        metric_definitions = parse_metric_definitions(metric_definitions_path)
        if metric_definitions:
            add_metric_definitions_page(pdf, metric_definitions)
        else:
            print("Warning: Metric definitions page could not be generated.")
        
        # Add Experiment Settings Page
        settings_data = load_experiment_settings(experiment_dir)
        if settings_data:
            add_settings_table_page(pdf, settings_data, os.path.basename(os.path.normpath(experiment_dir)))
        else:
            print("Warning: Experiment settings page could not be generated.")
        
        # Add Prediction Error Histograms
        plot_prediction_error_histograms(results, pdf)
        
        # Generate boxplots
        plot_boxplots(results, pdf)
        
        # Generate histograms
        plot_histograms(results, pdf)
    
        print("DEBUG: ===== CALLING plot_performance_comparison (contains regression) =====")
        plot_performance_comparison(results, pdf, ordered_schemes=['FCFS', 'LRTF', 'SRTF', 'NPS'])

    print(f"Report generated: {output_pdf}")
    return output_pdf

def plot_boxplots(results, pdf):
    """
    Create boxplots comparing metrics across priority schemes, colored by number of agents.
    """
    # Define the metrics to plot and their labels
    metrics_config = {
        'closed_avg_simulated_NPS': 'Simulated NPS',
        'closed_avg_simulated_throughput_time': 'Simulated Throughput Time',
        'closed_percent': 'Closed Cases %',
        'case_queued_percent': 'Cases Queued %',
        'closed_avg_initial_delay': 'Average Initial Delay'
    }
    
    # Create a boxplot for each metric
    for metric, label in metrics_config.items():
        if metric not in results.columns:
            print(f"Warning: Column {metric} not found in results")
            continue
            
        plt.figure(figsize=(12, 6))
        
        # Create boxplot
        sns.boxplot(
            data=results,
            x='F_priority_scheme',
            y=metric,
            hue='F_number_of_agents',
            palette='viridis'
        )
        
        plt.title(f'{label} by Priority Scheme and Number of Agents', fontsize=14)
        plt.xlabel('Priority Scheme', fontsize=12)
        plt.ylabel(label, fontsize=12)
        
        # Apply specific y-axis limits
        if metric == 'closed_percent' or metric == 'case_queued_percent':
            plt.ylim(0, 100)
        elif metric == 'closed_avg_initial_delay':
            current_ylim = plt.gca().get_ylim()
            plt.ylim(0, current_ylim[1]) # Set lower bound to 0, keep upper bound

        plt.xticks(rotation=45)
        plt.legend(title='Number of Agents', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        pdf.savefig()
        plt.close()

def plot_histograms(results, pdf):
    """
    Create histograms for specified metrics.
    """
    # Define the metrics to plot and their labels
    metrics_config = {
        'closed_avg_predicted_throughput_time': 'Predicted Throughput Time',
        'closed_avg_predicted_NPS': 'Predicted NPS',
        'closed_avg_predicted_NPS_priority': 'Predicted NPS Priority',
        'max_tracelen': 'Maximum Trace Length',
        'all_avg_initial_delay': 'Average Initial Delay'
    }
    
    # Create a histogram for each metric
    for metric, label in metrics_config.items():
        if metric not in results.columns:
            print(f"Warning: Column {metric} not found in results")
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        sns.histplot(
            data=results,
            x=metric,
            bins=30,
            kde=True
        )
        
        plt.title(f'Distribution of {label}', fontsize=14)
        plt.xlabel(label, fontsize=12)
        plt.ylabel('Count', fontsize=12)

        # Apply specific x-axis limits for relevant metrics
        if metric == 'all_avg_initial_delay':
            current_xlim = plt.gca().get_xlim()
            plt.xlim(0, current_xlim[1]) # Set lower bound to 0, keep upper bound

        plt.tight_layout()
        
        pdf.savefig()
        plt.close()

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
    fig = None  # Initialize fig to None
    try:
        formula_parts = [dep_var_col, "~"]
        if predictor_cols_numerical:
            formula_parts.append(" + ".join(predictor_cols_numerical))
        
        if predictor_col_categorical:
            if predictor_cols_numerical:
                formula_parts.append(f" + C({predictor_col_categorical})")
            else:
                formula_parts.append(f"C({predictor_col_categorical})")
                
        if len(formula_parts) == 2:
            print(f"INFO: No valid predictors found for {dep_var_label}. Skipping this regression plot.")
            return # Don't create a figure or try to save

        formula = "".join(formula_parts)

        results_df[dep_var_col] = pd.to_numeric(results_df[dep_var_col], errors='coerce')
        cols_for_na_check = [dep_var_col] + predictor_cols_numerical
        if predictor_col_categorical:
            cols_for_na_check.append(predictor_col_categorical)
        
        cleaned_df = results_df.dropna(subset=cols_for_na_check).copy()

        if cleaned_df.empty or cleaned_df[dep_var_col].nunique() < 2:
            raise ValueError("Not enough data or variance in dependent variable after cleaning.")

        valid_numerical_predictors = []
        for p_col in predictor_cols_numerical:
            if cleaned_df[p_col].nunique() > 1:
                valid_numerical_predictors.append(p_col)
            else:
                print(f"Warning: Predictor '{p_col}' has no variance after NA drop for DV '{dep_var_label}', removing from this regression.")
        
        formula_parts_cleaned = [dep_var_col, "~"]
        if valid_numerical_predictors:
            formula_parts_cleaned.append(" + ".join(valid_numerical_predictors))
        
        if predictor_col_categorical and cleaned_df[predictor_col_categorical].nunique() > 1:
            if valid_numerical_predictors:
                 formula_parts_cleaned.append(f" + C({predictor_col_categorical})")
            else:
                 formula_parts_cleaned.append(f"C({predictor_col_categorical})")
        elif predictor_col_categorical:
             print(f"Warning: Categorical predictor '{predictor_col_categorical}' has no variance after NA drop for DV '{dep_var_label}', removing from this regression.")

        if len(formula_parts_cleaned) == 2:
             raise ValueError("No valid predictors left after checking variance post NA drop.")
        
        formula_cleaned = "".join(formula_parts_cleaned)
        
        # Create figure just before populating
        fig = plt.figure(figsize=(11.7, 16.5)) # A4 size landscape
        plt.axis('off')

        print(f"DEBUG (plot_regression_analysis): Using formula: {formula_cleaned}")
        if not cleaned_df[[dep_var_col] + valid_numerical_predictors].empty:
            print(f"DEBUG (plot_regression_analysis): Cleaned DF head for {dep_var_label}:\\n{cleaned_df[[dep_var_col] + valid_numerical_predictors].head()}")
        else:
            print(f"DEBUG (plot_regression_analysis): Cleaned DF for {dep_var_label} is empty or predictors are missing after head().")

        model = smf.ols(formula_cleaned, data=cleaned_df).fit()
        summary_text = str(model.summary())
        
        print(f"DEBUG (plot_regression_analysis): Summary text (first 300 chars) for {dep_var_label}:\\n{summary_text[:300]}")
        
        # Update title with the formula actually used, now with improved formatting
        title_string = f"Regression Analysis for:\n{dep_var_label}\n\nFormula:\n{formula_cleaned}"
        plt.title(title_string, fontsize=12, y=1.0, loc='left', wrap=True) # y=1.0 might need adjustment with more lines
        # Adjust figure layout to make space for the multi-line title
        plt.subplots_adjust(top=0.85) # Reduce top to make space if title is long
        
        plt.text(0.01, 0.90 - (title_string.count('\n') * 0.05) , summary_text, family='monospace', fontsize=8, va='top', ha='left') # Adjust y of summary
        
        pdf_object.savefig(fig)

    except Exception as e:
        error_message = f"Could not perform regression for {dep_var_label}.\\nError: {str(e)}"
        print(error_message)
        # If fig was created, add error message to it, otherwise create a new one for the error
        if fig is None:
            fig = plt.figure(figsize=(11.7, 16.5))
            plt.axis('off')
        fig.clf() # Clear the figure in case it was partially populated
        plt.text(0.5, 0.5, error_message, ha='center', va='center', fontsize=10, color='red', wrap=True)
        if pdf_object is not None: # Ensure pdf_object is valid before saving
             pdf_object.savefig(fig)

    finally:
        if fig is not None:
            plt.close(fig) # Close the specific figure

def plot_performance_comparison(results, pdf, ordered_schemes):
    """
    Plot key performance metrics for comparison.
    """
    
    # Define metrics to plot, their y-axis labels, and titles
    metrics_to_plot_config = {
        'closed_avg_simulated_NPS': {
            'type': 'boxplot',
            'y_label': 'Average Simulated NPS (per run)',
            'title': 'Distribution of Average Simulated NPS (per run) by Priority Scheme'
        },
        'closed_avg_simulated_throughput_time': {
            'type': 'boxplot',
            'y_label': 'Average Simulated Throughput Time (h, per run)',
            'title': 'Distribution of Average Simulated Throughput Time (per run) by Priority Scheme'
        },
        'cases_closed': {
            'type': 'boxplot',
            'y_label': 'Closed Cases (per run)',
            'title': 'Distribution of Closed Cases (per run) by Priority Scheme'
        },
        'closed_avg_initial_delay': {
            'type': 'boxplot', # Changed from barplot
            'y_label': 'Average Initial Delay (h, per run)', # Updated label
            'title': 'Distribution of Average Initial Delay (per run) by Priority Scheme' # Updated title
        }
    }
    
    available_metrics_config = {
        metric: config for metric, config in metrics_to_plot_config.items() if metric in results.columns
    }

    palette = sns.color_palette("viridis", n_colors=len(ordered_schemes))
    scheme_colors = {scheme: palette[i] for i, scheme in enumerate(ordered_schemes)}

    for metric, config in available_metrics_config.items():
        plt.figure(figsize=(10, 6))
        
        if config['type'] == 'boxplot':
            sns.boxplot(
                x='F_priority_scheme',
                y=metric,
                data=results, # Use original results for boxplot per run
                order=ordered_schemes,
                palette="viridis",
                hue='F_number_of_agents',
                showfliers=False # Optionally hide fliers for cleaner plots
            )
            plt.legend(title='Number of Agents', bbox_to_anchor=(1.05, 1), loc='upper left')
        elif config['type'] == 'barplot':
            # For barplot, we show the mean of the per-run values
            plot_data = results.groupby('F_priority_scheme')[metric].mean().reindex(ordered_schemes).reset_index()
            sns.barplot(
                x='F_priority_scheme',
                y=metric,
                data=plot_data,
                order=ordered_schemes,
                palette=scheme_colors,
                errorbar=None # Changed from ci=None for newer seaborn versions
            )
        
        plt.title(config['title'], fontsize=16)
        plt.xlabel('Priority Scheme', fontsize=12)
        plt.ylabel(config['y_label'], fontsize=12)
        
        # Apply specific y-axis limits
        if metric == 'closed_avg_initial_delay':
            current_ylim = plt.gca().get_ylim()
            plt.ylim(0, current_ylim[1]) # Set lower bound to 0, keep upper bound
        elif metric == 'closed_percent': # Assuming closed_percent might be added here later
            plt.ylim(0,100)

        plt.xticks(rotation=45, ha="right") # Ensure x-axis labels are readable
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # --- Regression Analysis Section ---
    if results.empty:
        print("DEBUG: Skipping regression analysis as the results DataFrame is empty.")
    else:
        print("DEBUG: Entering regression analysis section.")
        predictor_cols_numerical = []
        potential_numerical_predictors = ['F_NPS_dist_bias', 'F_tNPS_wtime_effect_bias', 'F_number_of_agents']
        
        for col_name in potential_numerical_predictors:
            if col_name in results.columns and pd.to_numeric(results[col_name], errors='coerce').nunique() > 1:
                predictor_cols_numerical.append(col_name)
            elif col_name in results.columns:
                 print(f"Predictor '{col_name}' excluded from regression due to single unique value or non-numeric type.")


        categorical_predictor = 'F_priority_scheme' if 'F_priority_scheme' in results.columns and results['F_priority_scheme'].nunique() > 1 else None
        if 'F_priority_scheme' in results.columns and results['F_priority_scheme'].nunique() <=1 and categorical_predictor is None:
            print("DEBUG: Categorical predictor 'F_priority_scheme' excluded due to single unique value.")

        print(f"DEBUG: Numerical predictors selected: {predictor_cols_numerical}")
        print(f"DEBUG: Categorical predictor selected: {categorical_predictor}")

        dependent_vars_config = {
            'closed_avg_simulated_throughput_time': 'Simulated Throughput Time',
            'closed_percent': 'Proportion of Closed Cases (%)',
            'closed_avg_simulated_NPS': 'Simulated NPS'
        }

        min_obs_for_regression = len(predictor_cols_numerical) + (results[categorical_predictor].nunique() if categorical_predictor and categorical_predictor in results else 1) + 2 # Basic rule of thumb: k+1 obs, adding a small margin

        for dep_var_col, dep_var_label in dependent_vars_config.items():
            print(f"DEBUG: Processing regression for dependent variable: {dep_var_label} ({dep_var_col})")
            if dep_var_col in results.columns and pd.to_numeric(results[dep_var_col], errors='coerce').notnull().all():
                # Ensure the dependent variable itself has variance
                if results[dep_var_col].nunique() < 2:
                    print(f"DEBUG: Skipping regression for '{dep_var_label}' (column '{dep_var_col}') as it has less than 2 unique values.")
                    continue
                
                # Check if there are enough data points
                if len(results.dropna(subset=[dep_var_col] + predictor_cols_numerical + ([categorical_predictor] if categorical_predictor else []))) >= min_obs_for_regression:
                    plot_regression_analysis(results.copy(), pdf, dep_var_col, dep_var_label, predictor_cols_numerical, categorical_predictor)
                else:
                    print(f"DEBUG: Skipping regression for '{dep_var_label}' due to insufficient data points after potential NA drops (need at least {min_obs_for_regression}).")
            else:
                print(f"DEBUG: Skipping regression for '{dep_var_label}' as column '{dep_var_col}' is missing, not numeric, or all NaN.")
        
        print("DEBUG: Finished processing all dependent variables for regression.")
        # --- End of Regression Analysis Section ---

def plot_prediction_error_histograms(results, pdf):
    """
    Create histograms for MAE of TT and NPS prediction errors.
    """
    error_metrics_to_plot = {
        'mae_TT_pred_error': 'MAE of Throughput Time Prediction Error (hours)',
        'mae_NPS_pred_error': 'MAE of NPS Prediction Error'
    }

    for metric_col, label in error_metrics_to_plot.items():
        if metric_col not in results.columns or results[metric_col].isna().all():
            print(f"Warning: Column {metric_col} for histogram is missing or all NaN. Skipping histogram.")
            # Create a blank page with a message if preferred, or just skip
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'Data for "{label}" is not available or all NaN.', 
                     ha='center', va='center', fontsize=12, color='red')
            plt.title(f'Histogram for {label}', fontsize=14)
            plt.axis('off')
            pdf.savefig()
            plt.close()
            continue

        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=results,
            x=metric_col,
            bins=20, # Adjust bin count as needed
            kde=True,
            color='orange'
        )
        plt.title(f'Distribution of {label}', fontsize=14)
        plt.xlabel(label, fontsize=12)
        plt.ylabel('Frequency of Runs', fontsize=12)

        # Set lower x-axis limit to 0 for MAE plots
        current_xlim = plt.gca().get_xlim()
        plt.xlim(0, current_xlim[1]) # Set lower bound to 0, keep upper bound

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