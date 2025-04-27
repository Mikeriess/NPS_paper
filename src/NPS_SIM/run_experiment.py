# -*- coding: utf-8 -*-
"""
NPS Simulation Runner

This script runs a series of simulation experiments based on a design table.
Each experiment represents a different configuration of simulation parameters.
The script tracks progress, handles errors, and saves results for each run.

Created based on original run.py
"""

# Standard library imports
import time
import datetime
import logging
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Third-party imports
import pandas as pd 
import numpy as np
import sys

# Local imports
from algorithms.alg1_timeline_simulation import Run_simulation

# Default number of workers for parallel execution
MAX_WORKERS = mp.cpu_count()  # Maximum number of parallel processes (defaults to CPU count)

# Configure logging system
# Sets up logging with timestamp, log level, and message format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_experiments(results_dir: Path) -> pd.DataFrame:
    """
    Load and filter the design table containing experiment configurations.
    
    Args:
        results_dir (Path): Path to the directory containing the design table
        
    Returns:
        pd.DataFrame: DataFrame containing only experiments that haven't been run yet (Done == 0)
    
    Raises:
        FileNotFoundError: If the design table file doesn't exist
        Exception: For any other errors during loading
    """
    design_table_path = results_dir / "design_table.csv"
    
    try:
        # Check if design table exists
        if not design_table_path.exists():
            logger.error(f"Design table not found at {design_table_path}")
            raise FileNotFoundError(f"No design_table.csv found in {results_dir}. Please run generate_design.py first.")
        
        # Load the design table from CSV
        experiments = pd.read_csv(design_table_path)
        
        # Ensure datetime columns are string type
        datetime_columns = ["Started_at", "Finished_at"]
        for col in datetime_columns:
            if col in experiments.columns:
                experiments[col] = experiments[col].astype(str)
        
        # Filter to only include experiments that haven't been run
        return experiments.loc[experiments.Done == 0]
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error loading experiments: {str(e)}")
        raise


def create_results_directory(results_dir: Path, run: int) -> Path:
    """
    Create a directory for storing results of a specific run.
    
    Args:
        results_dir (Path): Path to the base results directory
        run (int): The run number/identifier
        
    Returns:
        Path: Path object pointing to the created directory
        
    Raises:
        Exception: If directory creation fails
    """
    run_dir = results_dir / str(run)
    try:
        # Create directory if it doesn't exist, including parent directories if needed
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    except Exception as e:
        logger.error(f"Error creating directory for run {run}: {str(e)}")
        raise


def save_results(run: int, evlog: pd.DataFrame, Case_DB: pd.DataFrame, run_dir: Path) -> None:
    """
    Save simulation results to CSV files.
    
    Args:
        run (int): The run number/identifier
        evlog (pd.DataFrame): Event log DataFrame to save
        Case_DB (pd.DataFrame): Case database DataFrame to save
        run_dir (Path): Directory where results should be saved
        
    Raises:
        Exception: If saving results fails
    """
    try:
        # Save event log and case database to separate CSV files
        evlog.to_csv(run_dir / f"{run}_log.csv", index=False)
        Case_DB.to_csv(run_dir / f"{run}_case_DB.csv", index=False)
    except Exception as e:
        logger.error(f"Error saving results for run {run}: {str(e)}")
        raise


def print_run_settings(run: int, settings: pd.Series) -> None:
    """
    Print the settings for a specific run in a readable format.
    
    Args:
        run (int): The run number/identifier
        settings (pd.Series): Series containing all simulation parameters
    """
    logger.info(f"\n\nSettings for Run {run}:\n\n")
    logger.info("=" * 50)
    for param, value in settings.items():
        if param not in ["Done", "Started_at", "Finished_at", "Simulation_duration_min"]:
            logger.info(f"{param}: {value}")
    logger.info("=" * 50)


def process_single_run(args: Tuple[int, pd.Series, Path]) -> Tuple[int, Dict[str, Any]]:
    """
    Process a single simulation run.
    This function is designed to be called in parallel.
    
    Args:
        args (Tuple[int, pd.Series, Path]): Tuple containing:
            - run number
            - settings Series
            - results directory Path
    
    Returns:
        Tuple[int, Dict[str, Any]]: Tuple containing:
            - run number
            - dictionary of results to update in the design table
    """
    run, settings, results_dir = args
    
    try:
        # Print settings for this run
        print_run_settings(run, settings)
        
        # Check if this run has already been completed
        run_dir = results_dir / str(run)
        log_file = run_dir / f"{run}_log.csv"
        
        if log_file.exists():
            logger.info(f"Run {run} already completed, skipping...")
            return run, {}
        
        # Create directory for this run's results
        run_dir = create_results_directory(results_dir, run)
        
        # Record start time
        start_time = datetime.datetime.now()
        
        # Set random seed for reproducibility - needs to be passed explicitly to each module
        seed_value = int(run)
        np.random.seed(seed_value)
        
        # Import the Run_simulation function directly to ensure it's accessible
        try:
            from algorithms.alg1_timeline_simulation import Run_simulation
        except ImportError:
            logger.error("Error importing Run_simulation. Check your module paths.")
            raise
        
        # Prepare simulation parameters
        sim_params = {
            "agents": int(settings["F_number_of_agents"]),
            "P_scheme": str(settings["F_priority_scheme"]),
            "ceiling": str(settings["F_hard_ceiling"]),
            "F_ceiling_value": float(settings["F_ceiling_value"]),
            "F_days": int(settings["F_days"]),
            "F_burn_in": int(settings["F_burn_in"]),
            "F_NPS_dist_bias": float(settings["F_NPS_dist_bias"]),
            "F_tNPS_wtime_effect_bias": float(settings["F_tNPS_wtime_effect_bias"]),
            "seed": seed_value,  # Explicitly pass the seed
            "startdate": datetime.datetime.fromisoformat(str(settings["startdate"])),
            "verbose": False,
            "results_dir": str(results_dir)  # Pass the results directory to the simulation
        }
        
        # Execute simulation with all parameters explicitly named
        logger.info(f"Starting simulation for run {run} with seed {seed_value}")
        evlog, Case_DB = Run_simulation(**sim_params)
        
        # Save simulation results
        save_results(run, evlog, Case_DB, run_dir)
        
        # Record end time and calculate duration
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate aggregate metrics from results
        results = {
            "Started_at": start_time.strftime('%Y-%m-%d %H:%M:%S'),
            "Finished_at": end_time.strftime('%Y-%m-%d %H:%M:%S'),
            "Simulation_duration_min": float(duration / 60),  # Ensure this is a float
            "Done": 1
        }
        
        # Add aggregated metrics for closed cases
        closed_cases = evlog[evlog.case_status == "closed"]
        if not closed_cases.empty:
            results["closed_avg_simulated_NPS"] = closed_cases.simulated_NPS.mean()
            results["closed_avg_simulated_throughput_time"] = closed_cases.simulated_throughput_time.mean()
            results["closed_avg_predicted_NPS"] = closed_cases.est_NPS.mean()
            results["closed_avg_predicted_throughput_time"] = closed_cases.est_throughput_time.mean()
            results["closed_avg_predicted_NPS_priority"] = closed_cases.est_NPS_priority.mean()
            results["closed_avg_initial_delay"] = closed_cases.initial_delay.mean()
            results["closed_avg_activity_start_delay"] = closed_cases.activity_start_delay.mean()
            results["closed_avg_duration_delayed"] = closed_cases.duration_delayed.mean()
        
        # Add metrics for all cases
        results["all_avg_simulated_NPS"] = evlog.simulated_NPS.mean()
        results["all_avg_simulated_throughput_time"] = evlog.simulated_throughput_time.mean()
        results["all_avg_predicted_NPS"] = evlog.est_NPS.mean()
        results["all_avg_predicted_throughput_time"] = evlog.est_throughput_time.mean()
        results["all_avg_predicted_NPS_priority"] = evlog.est_NPS_priority.mean()
        results["all_avg_initial_delay"] = evlog.initial_delay.mean()
        results["all_avg_activity_start_delay"] = evlog.activity_start_delay.mean()
        results["all_avg_duration_delayed"] = evlog.duration_delayed.mean()
        
        # Add case count metrics
        results["cases_arrived"] = len(Case_DB)
        results["cases_closed"] = len(Case_DB[Case_DB.status == "closed"])
        results["case_queued"] = len(Case_DB[Case_DB.case_queued == True])
        results["cases_assigned_at_end"] = len(Case_DB[Case_DB.case_currently_assigned == True])
        
        # Add trace length metrics
        case_event_counts = evlog.groupby('case_id').size()
        results["min_tracelen"] = case_event_counts.min() if not case_event_counts.empty else 0
        results["max_tracelen"] = case_event_counts.max() if not case_event_counts.empty else 0
        
        logger.info(f"Completed run {run} in {duration/60:.2f} minutes")
        return run, results
        
    except Exception as e:
        logger.error(f"Error processing run {run}: {str(e)}")
        # Print the full stack trace for debugging
        import traceback
        logger.error(traceback.format_exc())
        raise


def run_experiments(results_dir: Path, parallel: bool = True, max_workers: int = None):
    """
    Main execution function that orchestrates the simulation experiments.
    
    Args:
        results_dir (Path): Path to the directory containing design_table.csv and where results will be stored
        parallel (bool): Whether to run experiments in parallel
        max_workers (int): Maximum number of parallel processes. If None, uses CPU count.
    
    This function:
    1. Loads the experiment design table
    2. Iterates through each experiment
    3. Runs simulations for incomplete experiments
    4. Saves results and updates progress
    5. Handles errors and provides logging
    
    Raises:
        Exception: If a fatal error occurs during execution
    """
    design_table_path = results_dir / "design_table.csv"
    
    try:
        # Load experiment configurations
        experiments = load_experiments(results_dir)
        total_runs = len(experiments)
        
        if total_runs == 0:
            logger.info("No incomplete experiments found in the design table. All experiments are marked as done.")
            return
        
        if parallel:
            # Set number of workers
            num_workers = max_workers if max_workers is not None else MAX_WORKERS
            logger.info(f"Running {total_runs} experiments in parallel using {num_workers} workers")
            
            # Prepare arguments for parallel processing
            run_args = [(run, settings, results_dir) for run, settings in experiments.iterrows()]
            
            # Execute runs in parallel
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_run = {executor.submit(process_single_run, args): args[0] 
                                for args in run_args}
                
                # Collect results from all runs
                results = {}
                for future in as_completed(future_to_run):
                    run = future_to_run[future]
                    try:
                        run_num, run_results = future.result()
                        if run_results:  # Only include runs that were actually processed
                            results[run_num] = run_results
                    except Exception as e:
                        logger.error(f"Run {run} generated an exception: {str(e)}")
                
                # Update design table with all results in a single operation
                if results:
                    # Load full design table again to update
                    full_experiments = pd.read_csv(design_table_path)
                    for run, run_results in results.items():
                        for col, value in run_results.items():
                            full_experiments.at[run, col] = value
                    full_experiments.to_csv(design_table_path, index=False)
                    logger.info(f"Updated design table with results from {len(results)} runs")
        else:
            # Sequential execution
            logger.info(f"Running {total_runs} experiments sequentially")
            for idx, (run, settings) in enumerate(experiments.iterrows(), 1):
                logger.info(f"\nProcessing run {run} ({idx}/{total_runs})")
                try:
                    run_num, run_results = process_single_run((run, settings, results_dir))
                    if run_results:
                        # Load full design table again to update
                        full_experiments = pd.read_csv(design_table_path)
                        for col, value in run_results.items():
                            full_experiments.at[run, col] = value
                        full_experiments.to_csv(design_table_path, index=False)
                        logger.info(f"Updated design table for run {run}")
                except Exception as e:
                    logger.error(f"Error in run {run}: {str(e)}")
                    continue
            
    except Exception as e:
        logger.error(f"Fatal error in execution: {str(e)}")
        raise


def main():
    """Parse command line arguments and run experiments"""
    parser = argparse.ArgumentParser(description="Run NPS simulation experiments based on a design table.")
    parser.add_argument("--dest", type=str, required=True, 
                        help="Path to directory containing design_table.csv and where results will be stored")
    parser.add_argument("--parallel", action="store_true", default=True,
                        help="Run experiments in parallel (default: True)")
    parser.add_argument("--sequential", action="store_true", default=False,
                        help="Run experiments sequentially")  
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: CPU count)")
    
    args = parser.parse_args()
    
    # Convert destination to Path object
    results_dir = Path(args.dest)
    
    # Check if destination directory exists
    if not results_dir.exists():
        logger.error(f"Destination directory '{args.dest}' does not exist.")
        sys.exit(1)
    
    # Check if design_table.csv exists
    design_table_path = results_dir / "design_table.csv"
    if not design_table_path.exists():
        logger.error(f"No design_table.csv found in {args.dest}. Please run generate_design.py first.")
        sys.exit(1)
    
    # Determine whether to run in parallel or sequential mode
    parallel = not args.sequential if args.sequential else args.parallel
    
    # Run experiments
    run_experiments(results_dir, parallel=parallel, max_workers=args.workers)
    
    logger.info("All experiments completed successfully")


if __name__ == "__main__":
    main() 