# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 09:54:34 2022

@author: Mike
"""
import sys

print('parameters:', sys.argv)

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

#if the pipeline is run in batch mode: Arg1 "batch" arg2 batchno (int)
if sys.argv[1] == "batch":
    #load design table for this batch
    experiments = pd.read_csv("results/batches/Batch"+str(sys.argv[2])+"/experiments.csv")
    
    #remove runs that are not included in this batch
    #experiments = experiments.loc[experiments.Done == 0]



"""
Load simulation settings
"""


for run in experiments.RUN:
    
    # fix random seed for reproducibility
    seedvalue = int(run)
    np.random.seed(seedvalue)
    
    """
    Settings from experiments
    """
    curr_settings = experiments.loc[run]
    #curr_settings = experiments.loc[experiments.RUN == run].to_dict()
    print(curr_settings)
    # Get the number of the experiment
    RUN = run
    
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
        
        # store start time
        experiments.at[RUN, "Started_at"] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        
        path = "results/"+str(RUN)
        if os.path.isdir(path) == False:
            #create folder for the results
            os.mkdir(path)
        
        ####### Factors ######################        
        F_priority_scheme = curr_settings["F_priority_scheme"]
        F_number_of_agents = curr_settings["F_number_of_agents"]
        F_hard_ceiling = curr_settings["F_hard_ceiling"]
        days = curr_settings["days"]
        sim_start = datetime.datetime.fromisoformat(curr_settings["startdate"]) #.tolist()[0]
        ceiling_value = curr_settings["ceiling_value"]
        burn_in_period = curr_settings["burn_in"]
        NPS_bias = curr_settings["NPS_bias"]
        
        """ run the simulation """
        start_time = time.time()
        evlog, Case_DB = Run_simulation(agents=F_number_of_agents, 
                                                       P_scheme=F_priority_scheme, 
                                                       ceiling=F_hard_ceiling,
                                                       ceiling_value=ceiling_value,
                                                       D=days,
                                                       burn_in=burn_in_period,
                                                       seed=seedvalue,
                                                       startdate=sim_start,
                                                       NPS_bias=NPS_bias)
        
        evlog["RUN"] = RUN
        evlog.to_csv("results/"+str(RUN)+"/"+str(RUN)+"_log.csv",index=False)
        
        Case_DB["RUN"] = RUN        
        Case_DB["burn_in_period"] = Case_DB.arrival_q < burn_in_period-1
        Case_DB.to_csv("results/"+str(RUN)+"/"+str(RUN)+"_case_DB.csv",index=False)
        
        end_time = time.time()
        Time_sec = end_time - start_time
        
        ####### Results ######################
        
        # store finish time
        experiments.at[RUN, "Finished_at"] = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
        
    
        
        experiments.at[RUN, 'Simulation_duration_min'] = Time_sec/60
                        
        #store status
        experiments.at[RUN,"Done"] = 1
        

    #store status of experiments
    if sys.argv[1] == "batch":
        #update batch design table
        experiments.to_csv("results/batches/Batch"+str(sys.argv[2])+"/experiments.csv", index=False)