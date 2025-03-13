# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 15:14:04 2023

@author: Mike
"""
import pandas as pd
import os
import numpy as np

if os.path.isdir("results/batches") == False:
    os.mkdir("results/batches")
        
runs = len(pd.read_csv("results/design_table.csv"))

workers = 44

batches = np.linspace(0, runs, workers, endpoint=False).tolist()#[0:3]
batches = [int(i) for i in batches]

batch_size = int(runs/workers)

for batch in batches:
    
    # number of repetitions per batch
    start = int(batch)
    
    # number of current batch
    stop = start + batch_size
    
    # Get indexes this batch is responsible for
    ids = list(range(start,stop))
    
    # read design table
    experiments = pd.read_csv("results/design_table.csv")
    
    # Mark the design table
    experiments["batch_start"] = start
    
    # Create a folder
    if os.path.isdir("results/batches/Batch"+str(start)) == False:
        os.mkdir("results/batches/Batch"+str(start))
    
    for row in experiments.index:
        
        if row in(ids):
            #mark as not done, so it is performed in this batch
            experiments.loc[row, "Done"] = 0
        else:
            #mark as done, so it is not performed in this batch
            experiments.loc[row, "Done"] = 1

    # Overwrite the design table
    experiments.to_csv("results/batches/Batch"+str(start)+"/experiments.csv",index=False)
    
## estimated duration
est_runtime = 9

replications = 100
agents = 7
SLA = 2
Schemes = 4

minutes_per_worker = ((replications * agents * SLA * Schemes) * est_runtime)/workers

time_in_days = (minutes_per_worker/60)/24

print("Estimated runtime in days:",time_in_days)