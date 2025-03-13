# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 12:08:09 2023

@author: Mike
"""
import pandas as pd
import numpy as np
    
exp = pd.read_csv("results/design_table.csv")

exp = exp[['RUN','Done','F_priority_scheme', 'F_number_of_agents', 'F_hard_ceiling',
       'ceiling_value', 'burn_in', 'days', 'startdate', 'repetition']]

"""
Iterate over all runs in the batch
"""
for RUN in exp.RUN:
    print(RUN)
    
    # get the burn_in period
    burn_in = exp.loc[exp.RUN == RUN, "burn_in"].values[0]
    days = exp.loc[exp.RUN == RUN, "days"].values[0]
    burn_in = 30 ############################################# manual override <<<<<<<<

    # get the event-log
    evlog = pd.read_csv("results/"+str(RUN)+"/"+str(RUN)+"_log.csv")
    
    # get the case-db    
    Case_DB = pd.read_csv("results/"+str(RUN)+"/"+str(RUN)+"_case_DB.csv")

    """ Drop the burn_in period when calculating aggregated metrics"""

    if burn_in > 0:
        evlog = evlog.loc[evlog.case_arrival < days + burn_in]
        evlog = evlog.loc[evlog.case_arrival > burn_in]
        
        Case_DB = Case_DB.loc[Case_DB.arrival_q < days + burn_in]    
        Case_DB = Case_DB.loc[Case_DB.arrival_q > burn_in]    
    
    
    """ Look only at cases that did arrive in the period"""
    Case_DB = Case_DB.loc[Case_DB.case_currently_arrived == True]
    
    """ Calculate metrics on closed cases """        
    evlog_closed = evlog.loc[evlog["case_status"]=="closed"]
    
    # drop events from the eventlog with closed cases to get same weight of all cases
    evlog_closed = evlog_closed.drop_duplicates('case_id', keep='first')
    
    """
    ####################################################
    ## FULL SIMULATION PERIOD
    ####################################################
    """
        
    exp.at[RUN, 'closed_avg_simulated_NPS_response'] = np.mean(evlog_closed["simulated_NPS"])
    exp.at[RUN, 'closed_avg_simulated_throughput_time'] = np.mean(evlog_closed["simulated_throughput_time"])
    
    exp.at[RUN, 'closed_avg_predicted_throughput_time'] = np.mean(evlog_closed["est_throughput_time"])
    exp.at[RUN, 'closed_avg_predicted_NPS'] = np.mean(evlog_closed["est_NPS"])
    exp.at[RUN, 'closed_avg_predicted_NPS_priority'] = np.mean(evlog_closed["est_NPS_priority"])
    
    
    """
    ## Target variable: NPS
    """
    
    # drop events from the eventlog with closed cases
    evlog_closed = evlog_closed.drop_duplicates('case_id', keep='first')
    
    # calculate size of NPS segments
    n_promoters = len(evlog_closed.loc[evlog_closed.simulated_NPS >= 9])
    n_detractors = len(evlog_closed.loc[evlog_closed.simulated_NPS < 7])
    n_neutrals = len(evlog_closed) - (n_promoters + n_detractors)
    
    #NPS = % promoters - % detractors
    NPS = ((n_promoters/len(evlog_closed)) - (n_detractors/len(evlog_closed)))*100
    
    exp.at[RUN, 'n_promoters'] = n_promoters
    exp.at[RUN, 'n_detractors'] = n_detractors
    exp.at[RUN, 'n_neutrals'] = n_neutrals
    exp.at[RUN, 'simulated_NPS_org_level'] = NPS
        
    
    """ Calculate metrics on ALL cases """        
    evlog_all = evlog
    
    # drop events from the eventlog to get same weight of all cases
    evlog_all = evlog_all.drop_duplicates('case_id', keep='first')

    exp.at[RUN, 'all_avg_predicted_throughput_time'] = np.mean(evlog_all["est_throughput_time"])
    exp.at[RUN, 'all_avg_predicted_NPS'] = np.mean(evlog_all["est_NPS"])
    exp.at[RUN, 'all_avg_predicted_NPS_priority'] = np.mean(evlog_all["est_NPS_priority"])
    
    
    """ other metrics """        
    exp.at[RUN, 'cases_arrived'] = len(Case_DB)
    exp.at[RUN, 'cases_closed'] = len(evlog_closed)           
    
    exp.at[RUN, 'min_tracelen'] = np.min(evlog["event_no"])
    exp.at[RUN, 'max_tracelen'] = np.max(evlog["event_no"])


    ####### Timeseries results ######################
    
    # load a single run
    TS = pd.read_csv("results/"+str(RUN)+"/"+str(RUN)+"_timeseries.csv")
    
    if burn_in > 0:
        TS = TS.loc[TS.day < days + burn_in]
        TS = TS.loc[TS.day > burn_in]

    exp.at[RUN, 'avg_daily_queue_waitingtime'] = np.mean(TS["avg_current_queue_waitingtime"])
    exp.at[RUN, 'avg_n_cases_waiting_in_queue'] = np.mean(TS["n_cases_waiting_in_queue"])
    exp.at[RUN, 'avg_n_busy_agents'] = np.mean(TS["n_busy_agents"])
    
    

exp.to_csv("results/experiments.csv",index=False)