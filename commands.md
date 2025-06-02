
# Generate design
```bash
python src/NPS_SIM/generate_design.py --settings experiments/gridsearch_dynamic_tt/settings.json
```

# Run with all cpus
```bash
python src/NPS_SIM/run_experiment.py --dest experiments/gridsearch_dynamic_tt/
```

# Analyze/create pdf
```bash
python analysis/report_from_results.py --experiment experiments/gridsearch_dynamic_tt/
```

