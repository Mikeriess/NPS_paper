import subprocess
import argparse
import os
import sys
from pathlib import Path

def run_command(command, description):
    """Helper function to run a shell command and check for errors."""
    print(f"\nINFO: Starting: {description}")
    print(f"CMD: {' '.join(command)}")
    try:
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"STDOUT:\n{process.stdout}")
        if process.stderr:
            print(f"STDERR:\n{process.stderr}") # Print stderr even if command succeeds, for warnings
        print(f"INFO: Successfully completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed: {description}", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        print(f"Stdout:\n{e.stdout}", file=sys.stderr)
        print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print(f"ERROR: Command not found for '{description}'. Ensure scripts are in the correct paths and executable.", file=sys.stderr)
        print(f"Attempted command: {' '.join(command)}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the full NPS simulation experiment pipeline.")
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Path to the experiment directory (must contain settings.json).")
    parser.add_argument("--sequential", action="store_true", 
                        help="Run experiments sequentially instead of in parallel.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for running experiments (default: CPU count).")
    
    args = parser.parse_args()

    experiment_path = Path(args.experiment_dir).resolve() # Get absolute path
    settings_file = experiment_path / "settings.json"

    if not experiment_path.is_dir():
        print(f"ERROR: Experiment directory '{experiment_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not settings_file.is_file():
        print(f"ERROR: settings.json not found in '{experiment_path}'.", file=sys.stderr)
        sys.exit(1)

    # --- Step 1: Generate Design Table ---
    generate_design_cmd = [
        sys.executable, # Use the current python interpreter
        "src/NPS_SIM/generate_design.py",
        "--settings", str(settings_file)
    ]
    if not run_command(generate_design_cmd, "Generate Design Table"):
        sys.exit(1)

    # --- Step 2: Run Experiments ---
    run_experiments_cmd = [
        sys.executable,
        "src/NPS_SIM/run_experiment.py",
        "--dest", str(experiment_path)
    ]
    if args.sequential:
        run_experiments_cmd.append("--sequential")
    if args.workers is not None:
        run_experiments_cmd.extend(["--workers", str(args.workers)])
        
    if not run_command(run_experiments_cmd, "Run Experiments"):
        sys.exit(1)

    # --- Step 3: Generate Analysis Report ---
    generate_report_cmd = [
        sys.executable,
        "analysis/report_from_results.py",
        "--experiment", str(experiment_path)
    ]
    if not run_command(generate_report_cmd, "Generate Analysis Report"):
        sys.exit(1)

    print("\nINFO: Full experiment pipeline completed successfully!")

if __name__ == "__main__":
    main() 