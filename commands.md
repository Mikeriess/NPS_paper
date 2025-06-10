# Run docker
```bash
docker run -it --rm --gpus all -v "/storageHD/userHome/mikeriess/projects/NPS_paper:/NPS_paper" -w "/NPS_paper" nps_paper-simulation_runner:latest bash
```
# Generate design
```bash
python src/NPS_SIM/generate_design.py --settings experiments/contrast_experiment/settings.json
```

# Run with 40 cpus
```bash
python src/NPS_SIM/run_experiment.py --dest experiments/contrast_experiment/ --workers 40
```

# Analyze/create pdf
```bash
python analysis/report_from_results.py --experiment experiments/contrast_experiment/
```

