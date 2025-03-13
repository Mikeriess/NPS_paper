# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 21:34:52 2023

@author: Mike
"""

import pandas as pd
#pd.set_option('display.max_columns', None)
import numpy as np
import pickle
import mpmath
import os
import time
import datetime

import warnings
warnings.filterwarnings('ignore')

from algorithms.alg1_timeline_simulation import Run_simulation

"""
Load experiments
"""


experiments = pd.read_csv("results/design_table.csv")


for run in experiments.RUN:
    """
    New logic: Check if event-log already exist
    """
    import os.path
    
    log_name = "results/"+str(run)+"/"+str(run)+"_log.csv"
    
    # Bypass the experiment if it is already performed
    #if experiments.Done[RUN] == 0:
    if os.path.isfile(log_name) == False:
        
        print("================================"*3)
        print("RUN:",run)
        print("================================"*3)