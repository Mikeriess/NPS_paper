# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 20:25:36 2023

@author: Mike
"""

import pandas as pd
import os
import numpy as np
        
runs = len(pd.read_csv("results/design_table.csv"))

workers = 44

batches = np.linspace(0, runs, workers, endpoint=False).tolist()#[0:3]
batches = [int(i) for i in batches]

for batch in batches:
    #string = "cd NPS_SIM/Queue/P3_queue_prioritization \n"
    #string = string + "conda activate base \n"
    #string = string +"python run_experiments.py batch "+str(batch)+" \n\n"
    string ="!python run_experiments.py batch "+str(batch)+" \n\n"
    
    print(string)