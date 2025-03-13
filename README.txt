QUEUE PRIORITIZATION SIMULATION CODEBASE
========================================

This codebase contains a discrete event simulation framework for comparing different queue prioritization strategies in customer service scenarios. The code was used for the paper "Customer-service queuing based on predicted loyalty outcomes".

FOLDER STRUCTURE
---------------

1. algorithms/
   - Contains the core simulation algorithms
   - alg1_timeline_simulation.py: Main simulation engine
   - alg2_case_arrival.py: Handles case arrival generation
   - alg3_queue_management.py: Implements different queue prioritization strategies
   - alg4_case_assignment.py: Handles assignment of cases to agents
   - alg5_case_activities.py: Simulates case activities and their durations
   - alg6_finalize.py: Finalizes simulation and creates event logs

2. distributions/
   - Contains probability distributions used in the simulation
   - agents.py: Generates agent characteristics
   - p_vectors.py: Generates probability vectors for activities
   - tNPS.py: Simulates Net Promoter Score (NPS) responses

3. experiment/
   - Contains code for experimental design
   - DoE.py: Design of Experiments utilities
   - generate_experiments.py: Creates experimental designs

4. helpers/
   - Contains helper functions
   - run_batch_experiments.py: Runs batches of experiments
   - helper_functions.py: General utility functions

5. analysis/
   - Contains Jupyter notebooks for analyzing results
   - 1_convergence_behaviour.ipynb: Analyzes convergence of simulations
   - 2_Temporal_performance.ipynb: Analyzes temporal performance
   - 3_NPS_results.ipynb: Analyzes NPS results
   - Waiting_time_results.ipynb: Analyzes waiting time results

6. results/
   - Default directory for storing simulation results
   - Each experiment gets its own subdirectory
   - Contains event logs, case databases, and timeseries data

ROOT DIRECTORY SCRIPTS
---------------------

1. run_experiments.py
   - Main script to run experiments
   - Can be run in batch mode with arguments: python run_experiments.py batch <batch_number>
   - Loads experiment settings, runs simulations, and saves results

2. generate_experiments.py
   - Creates a design table with all experimental conditions
   - Defines factors like priority schemes, number of agents, etc.
   - Saves the design table to results/design_table.csv

3. split_to_batches.py
   - Splits the full experiment design into smaller batches for parallel execution
   - Creates batch folders in results/batches/
   - Useful for distributing work across multiple machines or cores

4. create_commands.py
   - Generates command strings to run each batch
   - Outputs commands that can be copied and pasted to run batches

5. collect_batches.py
   - Collects and aggregates results from all batches
   - Calculates metrics like NPS, throughput time, etc.
   - Creates a consolidated results file

6. check_results.py
   - Checks which experiments have been completed
   - Identifies missing experiment results

7. commands.py
   - Empty file, likely used to store generated commands

SIMULATION PARAMETERS
--------------------

The simulation includes several key parameters:
- F_priority_scheme: Queue discipline (FCFS, SRTF, LRTF, NPS)
- F_number_of_agents: Number of service agents
- F_hard_ceiling: Whether to use SLA constraints
- ceiling_value: Value for SLA ceiling
- burn_in: Burn-in period for simulation
- days: Number of days to simulate
- NPS_bias: Bias parameter for NPS simulation
- startdate: Start date for simulation

WORKFLOW
--------

1. Create experiments using generate_experiments.py
2. Split experiments into batches using split_to_batches.py
3. Run experiments using run_experiments.py (can use batch mode)
4. Collect and analyze results using collect_batches.py
5. Perform detailed analysis using notebooks in the analysis folder

For more information, refer to the paper or contact the authors. 