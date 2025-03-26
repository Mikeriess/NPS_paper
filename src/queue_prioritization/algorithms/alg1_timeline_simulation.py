# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 10:25:12 2022

@author: Mike
"""

import os

def Run_simulation(agents, P_scheme, ceiling, ceiling_value, D, burn_in, seed, startdate, NPS_bias, filename="Test_log.csv", verbose = False):
    
    import numpy as np
    np.random.seed(seed)
    import pandas as pd
    
    #from simulation.helper_functions import store_evlog, sim_generalized_lognormal
    from queue_prioritization.algorithms.alg2_case_arrival import CaseArrival#, pi_arr, pi_attr#, phi_arrival
    from queue_prioritization.algorithms.alg3_queue_management import QueueManagement
    from queue_prioritization.algorithms.alg4_case_assignment import CaseAssignment
    from queue_prioritization.algorithms.alg5_case_activities import CaseActivities
    from queue_prioritization.algorithms.alg6_finalize import store_evlog
    import datetime
    import time
    
    """
    Time series placeholder
    """
    timeseries = []
    
    """
    Generate the agents
    """
    
    from queue_prioritization.distributions.agents import generate_agents
    
    Psi = generate_agents(agents, seed)

    """
    Generate simulated activities using same seed
    """
    
    from queue_prioritization.distributions.p_vectors import generate_p_vectors
    
    p_vectors = generate_p_vectors(seed)

    """
    Days, case-buffer, case identifier
    """    
    
    L = []
    
    
    """
    Simulate case arrivals throughout the full period
    """
    i = 0
    counter = 0
    
    Theta, i = CaseArrival(i, D, P_scheme, startdate, seed=seed)
    
    
    """
    Generate case database: All cases throughout the simulation period
    """
    Case_DB = pd.DataFrame(pd.DataFrame({'theta_idx':list(range(0,len(Theta))), 
                                         'agent_idx':[None]*len(Theta), 
                                         'arrival_q':[v['q'] for v in Theta], 
                                         "est_throughputtime":[v['est_throughputtime'] for v in Theta], 
                                         "est_NPS":[v['est_NPS'] for v in Theta], 
                                         "est_NPS_priority":[v['est_NPS_priority'] for v in Theta],
                                         #FLAGS
                                         "case_currently_arrived":[False]*len(Theta),
                                         "case_currently_assigned":[False]*len(Theta),
                                         "case_queued":[False]*len(Theta),
                                         "status":["open"]*len(Theta)
                                         #"case_temporal_status":["Future_case"]*len(Theta)
                                         }))
    
    
    """ For each day in the simulation period """
    for d in list(range(0,D)):
        
        if verbose == True:
            print("#"*30)            
            print("day",str(d),"start")
            
        if d == 0:
            print(datetime.datetime.now(),":","day",str(d),"start")
            start_time = time.time()
        if d == 100:
            print(datetime.datetime.now(),":","day",str(d),"start")
        if d == 200:
            print(datetime.datetime.now(),":","day",str(d),"start")
        if d == 300:
            print(datetime.datetime.now(),":","day",str(d),"start")
        if d == 364:
            end_time = time.time()
            Time_sec = end_time - start_time
            print("Duration in minutes:",Time_sec/60)
        """ 
        daily snapshot of performance: measurement is at 00:00:01 when day begins
        """
        #Update status on newly arrived cases
        Case_DB["case_currently_arrived"] = Case_DB["arrival_q"] <= d+1
                
        # Process-level information
        all_cases = len(Case_DB)    
        future_cases = len(Case_DB.loc[np.where((Case_DB['case_currently_arrived']==False))])      
        currently_arrived = len(Case_DB.loc[np.where((Case_DB['case_currently_arrived']==True))])
        cases_assigned = len(Case_DB.loc[np.where(Case_DB['agent_idx'].notnull())]) # & Case_DB["case_temporal_status"] != "Future_case"))
        
        # Agent related information
        n_agents = len(pd.DataFrame(Psi))
        n_busy_agents = np.sum((pd.DataFrame(Psi)["status"] == "assigned to case")*1)            
        
        # Waiting times
        cases_waiting = Case_DB.loc[Case_DB["case_currently_arrived"] == True]
        cases_waiting = cases_waiting.loc[cases_waiting["case_queued"] == False]
        #cases_waiting["waiting_time"] = d - cases_waiting["arrival_q"]
                    
                
        cases_waiting_in_queue = Case_DB.loc[Case_DB["case_queued"] == True]        
        cases_waiting_in_queue["queue_waiting_time"] = d - cases_waiting_in_queue["arrival_q"]
                        
        morning_snapshot = {"day":d,
            
                            "n_cases_caseDB":len(Case_DB),
                             "n_cases_future_arrival":len(Case_DB.loc[np.where((Case_DB['case_currently_arrived']==False))]),
                             "n_cases_arrived":len(Case_DB.loc[np.where((Case_DB['case_currently_arrived']==True))]),
                             "n_cases_waiting_in_queue":len(Case_DB.loc[Case_DB["case_queued"] == True]),
                             
                             "n_cases_in_process":len(cases_waiting.loc[cases_waiting['case_currently_assigned']== True]),                                
                             
                             "n_cases_assigned_total":len(Case_DB.loc[np.where(Case_DB['agent_idx'].notnull())]),                             
                             "n_cases_closed_total":len(Case_DB.loc[Case_DB["status"] != "open"]),
                             
                             #agent related
                             "n_agents":n_agents,
                             "n_busy_agents":n_busy_agents,
                             
                             #cases waiting in the queue
                             "avg_current_queue_waitingtime":np.mean(cases_waiting_in_queue["queue_waiting_time"]),
                             "min_current_queue_waitingtime":np.min(cases_waiting_in_queue["queue_waiting_time"]),
                             "max_current_queue_waitingtime":np.max(cases_waiting_in_queue["queue_waiting_time"]),
                             "median_current_queue_waitingtime":np.median(cases_waiting_in_queue["queue_waiting_time"]),
                             
                             #cases being processed
                             # "avg_total_waitingtime":np.mean(cases_waiting["waiting_time"]),
                             # "min_total_waitingtime":np.min(cases_waiting["waiting_time"]),
                             # "max_total_waitingtime":np.max(cases_waiting["waiting_time"]),
                             # "median_total_waitingtime":np.median(cases_waiting["waiting_time"]),
                             }
                
        timeseries.append(morning_snapshot)
        
            
            
        """ Generate 15-minute increments"""
        minute_intervals = 15
        daily_increment = 1/((60*24)/minute_intervals) #### 15 minute intervals 
        daily_intervals = [daily_increment]*int(((60*24)/minute_intervals)) #12
        
        z = d
        
        """ Queue management, case assignment and activities"""
        for interval in daily_intervals:
            
            """ Update time by 15 minutes """
            z = z + interval
            
            """ Update case database with new cases that have arrived """
            # a case has arrived if its arrival time is before the end of the current window
            Case_DB["case_currently_arrived"] = Case_DB["arrival_q"] <= z
            
            # a case is in the queue if it has currently arrived, and no agent is assigned to it
            Case_DB["case_queued"] = Case_DB["case_currently_arrived"] & Case_DB['agent_idx'].isnull()
            
            """ If there are currently any cases in the queue or being processed """
            queue_size = np.sum(Case_DB["case_queued"]*1)
            open_cases = len(Case_DB.loc[np.where(Case_DB['agent_idx'].notnull())])
            
            if queue_size + open_cases > 0:
                
                """ Sort case order, based on queue priority scheme """        
                Case_DB = QueueManagement(Case_DB, 
                                          P_scheme,
                                          ceiling,
                                          ceiling_value, 
                                          time_interval = z,
                                          seed=seed)
                
                """ Assign idle agents to newly arrived cases """        
                Case_DB, Theta, Psi = CaseAssignment(Case_DB=Case_DB, 
                                                     Theta=Theta,
                                                     Psi=Psi, 
                                                     verbose=verbose, 
                                                     time_interval = z, 
                                                     seed=seed)
                
                """ Perform activities on the active cases """                    
                L, Case_DB, Theta, Psi, counter = CaseActivities(d, 
                                                        Case_DB, 
                                                        Theta, 
                                                        Psi, 
                                                        L, 
                                                        p_vectors=p_vectors,
                                                        seed=seed,
                                                        start_delays=True, 
                                                        end_delays=True, 
                                                        verbose=verbose, 
                                                        counter=counter)
                
            
    
    """
    Save an event-log
    """
    
    arrived_cases = len(Case_DB)
    
    if filename:  # Only try to save if a filename is provided
        # Make sure the directory exists
        directory = os.path.dirname(filename)
        if directory:  # Only create directories if there's a non-empty directory path
            os.makedirs(directory, exist_ok=True)
        
        # Save the event log
        evlog = store_evlog(L, P_scheme, agents, filename, burn_in, NPS_bias, seed)
        evlog.to_csv(filename, index=False)
    else:
        # Just create the event log without saving it
        evlog = store_evlog(L, P_scheme, agents, "", burn_in, NPS_bias, seed)
    
    # Generate daily timeseries snapshots for this simulation run
    timeseries = pd.DataFrame(timeseries)
    
    # if there are no customers in the queue to calculate metrics from
    # NaN values will be present: therefore we impute with zeros
    timeseries = timeseries.fillna(0)
    
    # Mark the burn_in period:
    timeseries["burn_in_period"] = timeseries.day < burn_in #".loc[timeseries.day > burn_in-1]
    
    # If the output is being written to a file
    if filename:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Save the log
        timeseries.to_csv(filename, index=False)
    
    return evlog, Case_DB


#evlog.est_NPS_priority