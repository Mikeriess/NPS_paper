# Queue Prioritization NPS Simulation Analysis Tool

This folder contains tools for analyzing the results of NPS queue prioritization simulation experiments. The main tool is `report.py`, which generates detailed visual reports from experiment results.

## `report.py` - Analysis Report Generator

### Overview

The `report.py` script analyzes simulation experiment results and generates a comprehensive PDF report with visualizations to help understand the effects of parameter changes on NPS scores and queue prioritization methods.

### Requirements

The script requires the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scipy

### Usage

```bash
python analysis/report.py --dest <experiment_directory>
```

Where `<experiment_directory>` is the path to an experiment results directory (e.g., `experiments/test1`).

#### Optional Arguments

- `--output <output_file>`: Specify a custom output file path for the PDF report. If not provided, the report will be saved in the experiment directory with an automatically generated name.

### Example

```bash
python analysis/report.py --dest experiments/test1
```

This will generate a PDF report in the `experiments/test1` directory with analysis of all the simulation runs in that experiment.

### Generated Report Contents

The PDF report includes the following visualizations and analyses:

1. **Title Page and Experiment Summary**
   - Overview of the experiment parameters
   - Summary statistics table for each prioritization method

2. **NPS Distribution Analysis**
   - Density plots showing the distribution of NPS scores for each prioritization method
   - Comparative distributions across different parameter combinations

3. **Parameter Effect Analysis**
   - Heatmaps showing how NPS bias and throughput time effect parameters impact:
     - Average NPS scores
     - Average throughput times

4. **Method Comparison**
   - Bar charts comparing NPS scores across different prioritization methods
   - Bar charts comparing throughput times across different prioritization methods
   - Radar charts for multi-metric comparison (when applicable)

5. **Sensitivity Analysis**
   - Line plots showing how NPS scores change with parameter changes
   - Threshold analysis identifying when the NPS method becomes superior to alternatives

## Report Interpretation

The generated report helps answer key research questions for Study 3:

1. **Parameter Effect Analysis**: How changes in NPS distribution bias and throughput time effect multiplier impact the distribution of NPS scores

2. **Sensitivity Thresholds**: At what point parameter changes cause meaningful shifts in the relative performance of prioritization methods

3. **Method Comparison**: How different queue prioritization strategies compare under various parameter settings

The report provides both visual insights and quantitative metrics to assess the robustness of the NPS-based queue prioritization approach relative to traditional methods (FCFS, LRTF, SRTF). 