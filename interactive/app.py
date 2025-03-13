# -*- coding: utf-8 -*-
"""
Flask application for running NPS queue simulation experiments interactively.
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
import pandas as pd
import numpy as np
import time
import datetime
import json

# Add parent directory to path so we can import from the main codebase
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import simulation functions
from experiment.DoE import build_full_fact, fix_label_values
from algorithms.alg1_timeline_simulation import Run_simulation

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/run_experiment', methods=['POST'])
def run_experiment():
    """Run experiments with the provided parameters for multiple priority schemes"""
    try:
        # Get parameters from request
        params = request.json
        
        # Ensure we have at least one priority scheme
        priority_schemes = params.get('priority_schemes', [])
        if not priority_schemes:
            return jsonify({
                'status': 'error',
                'message': 'At least one priority scheme must be selected'
            }), 400
        
        # Create results directory if it doesn't exist
        results_dir = "interactive_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Generate a unique run ID based on timestamp
        run_id = f"interactive_{int(time.time())}"
        run_dir = os.path.join(results_dir, run_id)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            
        # Log the experiment parameters
        with open(os.path.join(run_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)
        
        # Run the simulation for each priority scheme
        start_time = time.time()
        
        # Convert startdate string to datetime
        startdate = datetime.datetime.strptime(params['startdate'], '%Y-%m-%d')
        
        # Dictionary to store results for each scheme
        results = {}
        
        for scheme in priority_schemes:
            # Create a subdirectory for this scheme
            scheme_dir = os.path.join(run_dir, scheme)
            if not os.path.exists(scheme_dir):
                os.makedirs(scheme_dir)
            
            # Run the simulation with our patched function
            evlog, Case_DB = patched_run_simulation(
                agents=int(params['number_of_agents']),
                P_scheme=scheme,
                ceiling=params['hard_ceiling'],
                ceiling_value=float(params['ceiling_value']),
                D=int(params['days']),
                burn_in=int(params['burn_in']),
                seed=42,  # Fixed seed for reproducibility
                startdate=startdate,
                NPS_bias=float(params['nps_bias']),
                output_dir=scheme_dir  # Pass the scheme-specific directory
            )
            
            # Store results
            results[scheme] = {
                'evlog': evlog,
                'Case_DB': Case_DB
            }
        
        # Calculate overall metrics
        total_cases = sum(len(r['Case_DB']) for r in results.values())
        closed_cases = sum(len(r['evlog'][r['evlog']['case_status'] == 'closed']) for r in results.values())
        
        metrics = {
            'run_id': run_id,
            'execution_time': time.time() - start_time,
            'total_cases': total_cases,
            'closed_cases': closed_cases,
            'schemes': priority_schemes
        }
        
        return jsonify({
            'status': 'success',
            'message': 'Experiments completed successfully',
            'run_id': run_id,
            'metrics': metrics
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error running experiment: {str(e)}")
        print(error_details)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'details': error_details
        }), 500

@app.route('/api/experiments')
def list_experiments():
    """List all interactive experiments"""
    results_dir = "interactive_results"
    if not os.path.exists(results_dir):
        return jsonify([])
        
    experiments = []
    for run_id in os.listdir(results_dir):
        run_dir = os.path.join(results_dir, run_id)
        if os.path.isdir(run_dir) and os.path.exists(os.path.join(run_dir, "params.json")):
            with open(os.path.join(run_dir, "params.json"), "r") as f:
                params = json.load(f)
                
            experiments.append({
                'run_id': run_id,
                'timestamp': run_id.split('_')[1],
                'params': params
            })
    
    return jsonify(experiments)

@app.route('/experiment/<run_id>')
def experiment_results(run_id):
    """Display detailed results for a specific experiment with multiple schemes"""
    results_dir = "interactive_results"
    run_dir = os.path.join(results_dir, run_id)
    
    if not os.path.exists(run_dir):
        return "Experiment not found", 404
        
    # Load experiment parameters
    with open(os.path.join(run_dir, "params.json"), "r") as f:
        params = json.load(f)
    
    # Get the priority schemes
    priority_schemes = params.get('priority_schemes', [])
    
    # Dictionary to store results for each scheme
    scheme_results = {}
    
    for scheme in priority_schemes:
        scheme_dir = os.path.join(run_dir, scheme)
        
        if not os.path.exists(scheme_dir):
            continue
            
        # Check if result files exist
        log_file = os.path.join(scheme_dir, f"{scheme}_log.csv")
        case_db_file = os.path.join(scheme_dir, f"{scheme}_case_DB.csv")
        
        if not os.path.exists(log_file) or not os.path.exists(case_db_file):
            continue
        
        try:
            # Load event log and case database
            evlog = pd.read_csv(log_file)
            case_db = pd.read_csv(case_db_file)
            
            # Calculate metrics for this scheme
            metrics = calculate_experiment_metrics(evlog, case_db, params)
            
            # Generate plots for this scheme
            plots = generate_experiment_plots(evlog, case_db, params)
            
            scheme_results[scheme] = {
                'metrics': metrics,
                'plots': plots
            }
        except Exception as e:
            print(f"Error processing results for scheme {scheme}: {str(e)}")
    
    if not scheme_results:
        return render_template('experiment_error.html', 
                              run_id=run_id,
                              params=params,
                              error="No valid results found for any priority scheme.")
    
    # Generate comparison plots
    comparison_plots = generate_comparison_plots(scheme_results, params)
    
    return render_template('experiment_results.html', 
                          run_id=run_id,
                          params=params,
                          scheme_results=scheme_results,
                          comparison_plots=comparison_plots)

def calculate_experiment_metrics(evlog, case_db, params):
    """Calculate key metrics for the experiment"""
    # Filter closed cases
    closed_cases = evlog[evlog['case_status'] == 'closed']
    
    # Basic metrics
    metrics = {
        'total_cases': len(case_db),
        'closed_cases': len(closed_cases.drop_duplicates('case_id')),
        'avg_nps': closed_cases['simulated_NPS'].mean(),
        'avg_throughput_time': closed_cases['simulated_throughput_time'].mean(),
        'median_throughput_time': closed_cases['simulated_throughput_time'].median(),
        'min_throughput_time': closed_cases['simulated_throughput_time'].min(),
        'max_throughput_time': closed_cases['simulated_throughput_time'].max(),
    }
    
    # NPS distribution
    nps_counts = closed_cases['simulated_NPS'].value_counts().sort_index()
    promoters = nps_counts[nps_counts.index >= 9].sum()
    passives = nps_counts[(nps_counts.index >= 7) & (nps_counts.index <= 8)].sum()
    detractors = nps_counts[nps_counts.index <= 6].sum()
    
    total_responses = promoters + passives + detractors
    if total_responses > 0:
        nps_score = ((promoters / total_responses) - (detractors / total_responses)) * 100
    else:
        nps_score = 0
        
    metrics.update({
        'promoters': promoters,
        'passives': passives,
        'detractors': detractors,
        'nps_score': nps_score
    })
    
    # Time series data
    if 'case_arrival' in evlog.columns:
        # Group by day
        evlog['day'] = evlog['case_arrival'].astype(int)
        daily_metrics = evlog.groupby('day').agg({
            'case_id': 'nunique',
            'simulated_NPS': 'mean',
            'simulated_throughput_time': 'mean'
        }).reset_index()
        
        metrics['daily_metrics'] = daily_metrics.to_dict(orient='records')
    
    return metrics

def generate_experiment_plots(evlog, case_db, params):
    """Generate plots for the experiment results"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io
    import base64
    
    plots = {}
    
    # Filter closed cases
    closed_cases = evlog[evlog['case_status'] == 'closed']
    
    # 1. NPS Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='simulated_NPS', data=closed_cases, palette='viridis')
    plt.title('NPS Distribution')
    plt.xlabel('NPS Score')
    plt.ylabel('Count')
    
    # Save plot to a base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['nps_distribution'] = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    # 2. Throughput Time Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(closed_cases['simulated_throughput_time'], bins=30, kde=True)
    plt.title('Throughput Time Distribution')
    plt.xlabel('Throughput Time (days)')
    plt.ylabel('Count')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['throughput_distribution'] = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    # 3. Time Series: NPS over time
    if 'case_arrival' in evlog.columns:
        # Group by day
        evlog['day'] = evlog['case_arrival'].astype(int)
        daily_nps = evlog.groupby('day')['simulated_NPS'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(daily_nps['day'], daily_nps['simulated_NPS'])
        plt.title('Average NPS Over Time')
        plt.xlabel('Simulation Day')
        plt.ylabel('Average NPS')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plots['nps_time_series'] = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # 4. Time Series: Throughput Time over time
        daily_tt = evlog.groupby('day')['simulated_throughput_time'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(daily_tt['day'], daily_tt['simulated_throughput_time'])
        plt.title('Average Throughput Time Over Time')
        plt.xlabel('Simulation Day')
        plt.ylabel('Average Throughput Time (days)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plots['tt_time_series'] = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
    
    return plots

def patched_run_simulation(agents, P_scheme, ceiling, ceiling_value, D, burn_in, seed, startdate, NPS_bias, output_dir=None):
    """
    A wrapper around Run_simulation that redirects output files to a custom directory
    """
    import os
    import tempfile
    import shutil
    from algorithms.alg1_timeline_simulation import Run_simulation
    
    # Store absolute paths to avoid confusion
    if output_dir:
        output_dir_abs = os.path.abspath(output_dir)
    else:
        output_dir_abs = None
    
    # Create a temporary directory for the simulation to write to
    original_dir = os.getcwd()
    original_dir_abs = os.path.abspath(original_dir)
    
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Create the results directory structure that the simulation expects
        os.makedirs(os.path.join(temp_dir, "results"))
        os.makedirs(os.path.join(temp_dir, "results", str(seed)))
        
        # Change to the temp directory
        os.chdir(temp_dir)
        
        # Run the simulation
        evlog, Case_DB = Run_simulation(
            agents=agents,
            P_scheme=P_scheme,
            ceiling=ceiling,
            ceiling_value=ceiling_value,
            D=D,
            burn_in=burn_in,
            seed=seed,
            startdate=startdate,
            NPS_bias=NPS_bias
        )
        
        # Return to original directory before saving files
        os.chdir(original_dir_abs)
        
        # Save the dataframes directly to our output directory
        if output_dir_abs:
            # Make sure the output directory exists
            os.makedirs(output_dir_abs, exist_ok=True)
            
            # Get the run ID from the directory name
            run_id = os.path.basename(output_dir_abs)
            
            # Save the event log and case database
            evlog_file = os.path.join(output_dir_abs, f"{run_id}_log.csv")
            case_db_file = os.path.join(output_dir_abs, f"{run_id}_case_DB.csv")
            
            print(f"Saving event log to: {evlog_file}")
            print(f"Saving case DB to: {case_db_file}")
            
            evlog.to_csv(evlog_file, index=False)
            Case_DB.to_csv(case_db_file, index=False)
            
            # Try to find and save the timeseries data if it exists
            timeseries_path = os.path.join(temp_dir, "results", str(seed), f"{seed}_timeseries.csv")
            if os.path.exists(timeseries_path):
                timeseries_df = pd.read_csv(timeseries_path)
                timeseries_file = os.path.join(output_dir_abs, f"{run_id}_timeseries.csv")
                print(f"Saving timeseries to: {timeseries_file}")
                timeseries_df.to_csv(timeseries_file, index=False)
        
        return evlog, Case_DB
    
    except Exception as e:
        print(f"Error in patched_run_simulation: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise
    
    finally:
        # Make sure we're back in the original directory
        if os.getcwd() != original_dir_abs:
            os.chdir(original_dir_abs)
        
        # Clean up the temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not remove temporary directory {temp_dir}: {str(e)}")

def generate_comparison_plots(scheme_results, params):
    """Generate plots comparing different priority schemes"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io
    import base64
    import numpy as np
    
    plots = {}
    
    # 1. NPS Score Comparison
    plt.figure(figsize=(10, 6))
    schemes = list(scheme_results.keys())
    nps_scores = [scheme_results[scheme]['metrics']['nps_score'] for scheme in schemes]
    
    bars = plt.bar(schemes, nps_scores, color=sns.color_palette('viridis', len(schemes)))
    plt.title('NPS Score Comparison')
    plt.xlabel('Priority Scheme')
    plt.ylabel('NPS Score')
    plt.ylim(-100, 100)  # NPS ranges from -100 to 100
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:.1f}',
                 ha='center', va='bottom', rotation=0)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['nps_comparison'] = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    # 2. Throughput Time Comparison
    plt.figure(figsize=(10, 6))
    avg_throughput = [scheme_results[scheme]['metrics']['avg_throughput_time'] for scheme in schemes]
    
    bars = plt.bar(schemes, avg_throughput, color=sns.color_palette('viridis', len(schemes)))
    plt.title('Average Throughput Time Comparison')
    plt.xlabel('Priority Scheme')
    plt.ylabel('Average Throughput Time (days)')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.1f}',
                 ha='center', va='bottom', rotation=0)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['throughput_comparison'] = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    # 3. Closed Cases Comparison
    plt.figure(figsize=(10, 6))
    closed_cases = [scheme_results[scheme]['metrics']['closed_cases'] for scheme in schemes]
    
    bars = plt.bar(schemes, closed_cases, color=sns.color_palette('viridis', len(schemes)))
    plt.title('Closed Cases Comparison')
    plt.xlabel('Priority Scheme')
    plt.ylabel('Number of Closed Cases')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{int(height)}',
                 ha='center', va='bottom', rotation=0)
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plots['closed_cases_comparison'] = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    return plots

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 