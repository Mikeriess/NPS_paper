# Queue Prioritization Simulation

This package contains a discrete event simulation framework for comparing different queue prioritization strategies in customer service scenarios. The code was used for the paper "Customer-service queuing based on predicted loyalty outcomes".

## Installation

### Using Conda (Recommended for Windows)

```bash
# Clone the repository
git clone https://github.com/your-username/queue_prioritization.git
cd queue_prioritization

# Create and activate the conda environment
conda env create -f environment.yml
conda activate queue-sim
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/your-username/queue_prioritization.git
cd queue_prioritization

# Install the package
pip install -e .
```

## Usage

### Web Interface

To launch the web interface:

```bash
# Using the console script
queue-ui

# Or using the provided run script
python run_ui.py
```

The web interface allows you to:
- Configure and run simulations with different queue prioritization strategies
- Compare results across different strategies
- Visualize key metrics including NPS, throughput time, and more

### Programmatic Usage

You can also use the package programmatically:

```python
from queue_prioritization.algorithms.alg1_timeline_simulation import Run_simulation

# Run a simulation
evlog, Case_DB = Run_simulation(
    agents=10,  # Number of agents
    P_scheme="FCFS",  # Prioritization scheme (FCFS, SRTF, LRTF, NPS)
    ceiling=True,  # Use ceiling for SLA
    ceiling_value=30,  # Ceiling value for SLA
    D=30,  # Simulation days
    burn_in=10,  # Burn-in period
    seed=42,  # Random seed
    startdate="2022-01-01",  # Start date
    NPS_bias=0.5  # NPS bias
)
```

## Available Queue Disciplines

The following queue disciplines are implemented:

- **FCFS** (First-Come-First-Served): Cases are processed in order of arrival
- **SRTF** (Shortest Remaining Time First): Cases with shortest estimated throughput time are prioritized
- **LRTF** (Longest Remaining Time First): Cases with longest estimated throughput time are prioritized
- **NPS** (Net Promoter Score based): Cases are prioritized based on their estimated NPS impact

## Project Structure

- **src/queue_prioritization/algorithms/**: Core simulation algorithms
- **src/queue_prioritization/distributions/**: Probability distributions
- **src/queue_prioritization/experiment/**: Experimental design utilities
- **src/queue_prioritization/helpers/**: Helper functions
- **src/queue_prioritization/analysis/**: Analysis tools
- **src/queue_prioritization/web/**: Web interface
- **experiments/**: Results directory for simulations

## Acknowledegments

This codebase was developed for academic research on queue prioritization strategies in customer service scenarios. 