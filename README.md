# NPS Simulation Experiments

This repository contains a simulation framework for studying the effects of different queue prioritization methods on Net Promoter Score (NPS) outcomes.

## Overview

The simulation compares different queue prioritization methods:

- First Come, First Served (FCFS)
- Shortest Remaining Time First (SRTF)
- Longest Remaining Time First (LRTF)
- NPS-based prioritization (NPS)

The framework allows manipulation of parameters in the simulated NPS calculation to understand the sensitivity of results across different prioritization methods.

## Requirements

- Python 3.8 or higher
- Required packages: pandas, numpy

## Experiment Workflow

The experiment workflow consists of two main steps:

1. Generating a design table that defines the experiment configurations
2. Running the simulations based on the design table

## Step 1: Generate Design Table

The design table specifies the parameters for each simulation run using a full factorial design.

### Using JSON Configuration

1. Create a JSON settings file with the parameters to vary in your experiment:

```json
{
    "F_priority_scheme": ["NPS", "SRTF", "LRTF", "FCFS"],
    "F_number_of_agents": [3],
    "F_hard_ceiling": ["NONE"],
    "F_ceiling_value": [2.5],
    "F_burn_in": [0],
    "F_days": [50],
    "F_NPS_dist_bias": [-2, -1, -0.5, 0, 0.5, 1, 2],
    "F_tNPS_wtime_effect_bias": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "startdate": ["2018-07-01"],
    "repetition": [0]
}
```

2. Create a directory for your experiment:

```bash
mkdir -p experiments/experiment_name
```

3. Save your JSON settings file in the experiment directory:

```bash
cp your_settings.json experiments/experiment_name/settings.json
```

4. Generate the design table using the settings file:

```bash
python src/NPS_SIM/generate_design.py --settings experiments/experiment_name/settings.json
```

This will create a `design_table.csv` file in the same directory as your settings file. The design table contains all combinations of the parameter values specified in the JSON file, along with placeholders for result metrics.

## Step 2: Run Experiments

Once you have generated a design table, you can run the experiments using the `run_experiment.py` script:

```bash
python src/NPS_SIM/run_experiment.py --dest experiments/experiment_name
```

### Command Line Options

- `--dest`: (Required) Path to the directory containing the design table and where results will be stored
- `--parallel`: (Default: True) Run experiments in parallel
- `--sequential`: Run experiments sequentially instead of in parallel
- `--workers`: Number of parallel workers (default: CPU count)

Example with specific number of workers:

```bash
python src/NPS_SIM/run_experiment.py --dest experiments/experiment_name --workers 4
```

Example running sequentially:

```bash
python src/NPS_SIM/run_experiment.py --dest experiments/experiment_name --sequential
```

## Results

After running the experiments, the following files will be generated in your experiment directory:

1. Updated `design_table.csv` with results for each experiment
2. Subdirectories for each run (named by run number) containing:
   - `{run}_log.csv`: Detailed event log for the simulation
   - `{run}_case_DB.csv`: Database of all cases in the simulation
   - `{run}_timeseries.csv`: Time series data from the simulation

## Complete Example Workflow

```bash
# Create experiment directory
mkdir -p experiments/test1

# Create a settings file (or copy an existing one)
cp example.json experiments/test1/settings.json

# Generate the design table
python src/NPS_SIM/generate_design.py --settings experiments/test1/settings.json

# Run the experiments
python src/NPS_SIM/run_experiment.py --dest experiments/test1

# Check the results
ls -la experiments/test1
```

## Additional Information

- Experiments marked as "Done" in the design table will be skipped if you re-run the experiments.
- If the simulation is interrupted, you can resume it by running the same command again.
- For large experiment designs, consider running with fewer parameter combinations first to estimate runtime.
- Manipulating the `F_NPS_dist_bias` and `F_tNPS_wtime_effect_bias` parameters allows exploring different relationships between throughput time and NPS scores.

## Parameter Descriptions

- `F_priority_scheme`: Queue prioritization method
- `F_number_of_agents`: Number of agents (resources) in the simulation
- `F_hard_ceiling`: Whether to use hard ceiling for prioritization
- `F_ceiling_value`: Value of the ceiling if used
- `F_burn_in`: Burn-in period length in days
- `F_days`: Number of days to simulate
- `F_NPS_dist_bias`: Additive bias to the intercept of the Gamma regression for NPS
- `F_tNPS_wtime_effect_bias`: Multiplicative scaling factor for the log-transformed throughput time effect on NPS
- `startdate`: Simulation start date
- `repetition`: Repetition number for the experiment 