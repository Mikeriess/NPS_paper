# Run docker
```bash
docker run -it --rm --gpus all -v "/storageHD/userHome/mikeriess/projects/NPS_paper:/NPS_paper" -w "/NPS_paper" nps_paper-simulation_runner:latest bash
```
# Generate design
```bash
python src/NPS_SIM/generate_design.py --settings experiments/test_lasso_wtimebias/settings.json
```

# Run with all cpus
```bash
python src/NPS_SIM/run_experiment.py --dest experiments/test_lasso_wtimebias/
```

# Analyze/create pdf
```bash
python analysis/report_from_results.py --experiment experiments/test_lasso_wtimebias/
```

