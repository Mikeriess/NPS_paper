# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 13:49:18 2022

@author: Mike
"""

#### Generate a folder for the experiments

import os
import time
import numpy as np
import pandas as pd
import shutil
import sys

from queue_prioritization.algorithms.alg1_timeline_simulation import Run_simulation

# Define batch parameters
batch_number = 5

# Define paths
batch_name = "batch" + str(batch_number)
root_path = os.path.join(os.getcwd(), "experiments")
code_root_path = ""

# Create batch directory
batchpath = os.path.join(root_path, batch_name) 

def setup_batch_directory():
    """Set up the directory structure for the batch"""
    
    # Make the folder for the batch of experiments
    if not os.path.exists(batchpath):
        os.makedirs(batchpath) 

    # Make the folder to save the eventlogs
    if not os.path.exists(os.path.join(batchpath, "evlogs")):
        os.makedirs(os.path.join(batchpath, "evlogs")) 

    # Copy experiment files
    shutil.copyfile(os.path.join(root_path, "experiments.csv"), 
                    os.path.join(batchpath, "experiments.csv"))

    shutil.copyfile(os.path.join(root_path, "Experiment_Settings.npy"), 
                    os.path.join(batchpath, "Experiment_Settings.npy"))

def run_batch():
    """Run all experiments in the batch"""
    
    # Set up the directory
    setup_batch_directory()
    
    # Set the workdir
    original_dir = os.getcwd()
    os.chdir(batchpath)
    
    try:
        # Load up the experiments
        experiments = pd.read_csv("experiments.csv")
        experiments.index = experiments.RUN.values
        experiment_list = experiments.RUN.values
        
        # Run experiments
        for experiment_i in experiment_list:
            print("="*60)
            print("Starting experiment: ", experiment_i)
            
            # Run experiment
            # (call to run_experiment function or integrated code)
            
        # Store results
        experiments.to_csv("experiments.csv", index=False)
    
    finally:
        # Make sure we return to the original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_batch_directory()
    else:
        run_batch()
