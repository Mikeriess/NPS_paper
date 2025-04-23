# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 13:49:18 2022

@author: Mike
"""

#### Generate a folder for the experiments

batch_number = 5

batch_name = "batch" + str(batch_number)
root_path = "results/"
code_root_path = ""
#root_path = "E:/CRM_paper/"
#code_root_path = "E:/CRM_paper/Agent_based_evlog_simulator-master/Agent_based_evlog_simulator-master/"

# importing os module 
import os 
  
#Experiment
batchpath = os.path.join(root_path, batch_name) 

# Make the folder for the batch of experiments
if not os.path.exists(batchpath):
    os.mkdir(batchpath) 

# Make the folder to save the eventlogs
if not os.path.exists(os.path.join(batchpath, "evlogs")):
    os.mkdir(os.path.join(batchpath, "evlogs")) 

import shutil

shutil.copyfile(os.path.join(root_path, "experiments.csv"), 
                os.path.join(batchpath, "experiments.csv"))

shutil.copyfile(os.path.join(root_path, "Experiment_Settings.npy"), 
                os.path.join(batchpath, "Experiment_Settings.npy"))

file_list = ["helper_functions.py",
             "Alg1_timeline_simulation.py",
             "Alg2_case_arrival.py",
             "Alg3_queue_management.py",
             "Alg4_case_assignment.py",
             "Alg5_case_activities.py"]

for filenm in file_list:
    shutil.copyfile(os.path.join(code_root_path, filenm), 
                os.path.join(batchpath, filenm))
    
# for reading the experiments table
#experiments_dest = os.path.join(root_path, "../experiments.csv")

import time
import numpy as np
import pandas as pd

from algorithms.alg1_timeline_simulation import Run_simulation


#########################################

# Set the workdir
os.chdir(batchpath)

#########################################

# Load up the experiments
experiments = pd.read_csv("experiments.csv")
experiments.index = experiments.RUN.values
experiment_list = experiments.RUN.values

#Loop through all experiments in the initial list
for experiment_i in experiment_list:
    print("================================"*3)
    print("Starting experiment: ",experiment_i)
        
    # Load up the experiments for potential updates:
    experiments = pd.read_csv("experiments.csv")
    experiments.index = experiments.RUN.values
    
    # Load the levels of the experiment
    Experiment_Settings = np.load('Experiment_Settings.npy',allow_pickle='TRUE').item()
    
    # Get the number of the experiment
    RUN = experiments.RUN[experiment_i]
    
    # Bypass the experiment if it is already performed
    if experiments.Done[experiment_i] == 0:
        
        
        ####### Factors ######################
        # Convert the factors into the original level values
        
        F_priority_scheme = Experiment_Settings["F_priority_scheme"][int(experiments.F_priority_scheme[RUN])]
        F_number_of_agents = int(experiments.F_number_of_agents[RUN])
        days = int(experiments.days[RUN])
        
        start_time = time.time()
        
        seed_value = RUN + int(batch_number*1000)
        print(seed_value)
        
        """ run the simulation """
        evlog, arrived_cases, case_db = Run_simulation(agents=F_number_of_agents, 
                       P_scheme=F_priority_scheme, 
                       D = days, # to 01/01 2020 or 01/01/2019
                       seed = seed_value)
        
        evlog["RUN"] = RUN
        evlog.to_csv("evlogs/"+str(RUN)+"_log.csv",index=False)
        
        case_db["RUN"] = RUN
        case_db.to_csv("evlogs/"+str(RUN)+"_case_db.csv",index=False)
        
        end_time = time.time()
        Time_sec = end_time - start_time
        
       ####### Results ######################
        
        """
        avg_est_NPS
        avg_est_throughput_time
        avg_est_NPS_priority
        min_tracelen
        max_tracelen
        avg_initial_delay
        avg_activity_start_delay
        avg_duration_delayed
        
        """
        
        """ Only calculate metrics on closed cases """
        
        evlog_closed = evlog.loc[evlog["case_status"]=="closed"]
        evlog_all = evlog
        
        experiments.at[RUN, 'avg_actual_NPS'] = np.mean(evlog_closed["actual_NPS"])
        experiments.at[RUN, 'avg_actual_throughput_time'] = np.mean(evlog_closed["actual_throughput_time"])
        
        experiments.at[RUN, 'avg_predicted_NPS'] = np.mean(evlog_closed["est_NPS"])
        experiments.at[RUN, 'avg_predicted_throughput_time'] = np.mean(evlog_closed["est_throughput_time"])
        
        experiments.at[RUN, 'avg_predicted_NPS_priority'] = np.mean(evlog_closed["est_NPS_priority"])
        
        """ other metrics """        
        
        experiments.at[RUN, 'cases_arrived'] = arrived_cases
        experiments.at[RUN, 'cases_closed'] = evlog.loc[evlog["case_status"]=="closed"]['case_id'].nunique()
        experiments.at[RUN, 'cases_assigned'] = evlog_all['case_id'].nunique()
        
        experiments.at[RUN, 'min_tracelen'] = np.min(evlog_closed["event_no"])
        experiments.at[RUN, 'max_tracelen'] = np.max(evlog_closed["event_no"])
        experiments.at[RUN, 'avg_initial_delay'] = np.mean(evlog_closed["initial_delay"])
        experiments.at[RUN, 'avg_activity_start_delay'] = np.mean(evlog_closed["activity_start_delay"])
        experiments.at[RUN, 'avg_duration_delayed'] = np.mean(evlog_closed["duration_delayed"])
        experiments.at[RUN, 'Simulation_duration_min'] = Time_sec/60
        
        #flag that experiment is done
        experiments.at[RUN, 'Done'] = 1
        
        #Save the seed
        experiments.at[RUN, "seed"] = seed_value
        experiments["batch_number"] = batch_number
        
        # Lookup the values from the npy file
        variables = ["F_priority_scheme"]
        
       
                
        experiments.to_csv("experiments.csv",index=False)
        
        ######################################
    
"""        
if 'Name_fix' not in experiments:
           experiments["Name_fix"] = 0
           
           for variable in variables:
               print(variable)
               for RUN in experiments.RUN:
                   print(RUN)
                   idx = int(experiments[variable].loc[RUN])
                   print(idx)
                   value = Experiment_Settings[variable][idx]
                   experiments[variable].loc[RUN] = value
                   print(experiments[variable].loc[RUN])
               
           experiments["Name_fix"] = 1
           print(experiments)
"""
    
# Store results
experiments.to_csv("experiments.csv",index=False)
#experiments.to_csv("experiments"+batch_name+".csv",index=False)
