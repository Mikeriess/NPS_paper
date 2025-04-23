# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 10:25:12 2022

@author: Mike
"""

def Run_simulation(agents, 
                   P_scheme, 
                   ceiling, 
                   F_ceiling_value, 
                   F_days, 
                   F_burn_in, 
                   seed, 
                   startdate, 
                   F_NPS_dist_bias,
                   F_tNPS_wtime_effect_bias=1.0,
                   filename="Test_log.csv", 
                   verbose = False):
    """
    Run a simulation with the given parameters.
    
    Parameters:
    -----------
    agents : int
        Number of agents in the simulation
    P_scheme : str
        Priority scheme to use (FCFS, SIRO, SRTF, LRTF, NPS)
    ceiling : str
        Type of ceiling (NO, SLA)
    F_ceiling_value : float
        Value for the ceiling if using SLA
    F_days : int
        Number of days to simulate
    F_burn_in : int
        Number of days for burn-in period
    seed : int
        Random seed for reproducibility
    startdate : datetime
        Start date of the simulation
    F_NPS_dist_bias : float
        Bias to add to the NPS distribution
    F_tNPS_wtime_effect_bias : float
        Multiplier for the effect of waiting time on NPS
    filename : str
        Name of the output file
    verbose : bool
        Whether to print detailed output
        
    Returns:
    --------
    evlog : DataFrame
        Event log of all activities
    Case_DB : DataFrame
        Database of all cases
    """
    
    import numpy as np
    np.random.seed(seed)
    import pandas as pd
    
    #from simulation.helper_functions import store_evlog, sim_generalized_lognormal
    from algorithms.alg2_case_arrival import CaseArrival
    from algorithms.alg3_queue_management import QueueManagement
    from algorithms.alg4_case_assignment import CaseAssignment
    from algorithms.alg5_case_activities import CaseActivities
    
    # Define a local store_evlog function instead of importing it
    def store_evlog(L, P_scheme, agents, filedest, F_burn_in, F_NPS_dist_bias, seed, F_tNPS_wtime_effect_bias):
        import pandas as pd
        import numpy as np
        import datetime
        
        from distributions.tNPS import simulate_NPS

        DF_list = []
        
        # Step counter for random values seed
        step = seed
        
        for case in L:
            # Update step counter used for seed, each case gets its unique seed
            step = step + 1
            
            # Trace length 
            length = len(case["a"])
            
            # Mark the burn_in period cases
            case["burn_in"] = case["q"] < F_burn_in-1
                        
            if length > 0:
                # Calculate throughput time (incl. queue waiting time)
                throughput_time = np.max(case["t_end"])-case["q"]
                
                # Simulate NPS score after case closure
                NPS_simulated, NPS_priority_simulated = simulate_NPS(
                    case_topicidx=case["c_topic"], 
                    y=throughput_time, 
                    seed=step, 
                    F_NPS_dist_bias=F_NPS_dist_bias,
                    F_tNPS_wtime_effect_bias=F_tNPS_wtime_effect_bias
                )
                
                # Generate a dataframe
                res_i = pd.DataFrame({"case_id":[case["i"]]*length,
                                    
                                    "case_arrival":[case["q"]]*length,
                                    "burn_in_period":[case["burn_in"]]*length,
                                    
                                    "case_arrival_dt":[case["q_dt"].strftime('%m/%d/%Y %H:%M:%S')]*length,
                                    
                                    "case_assigned":[case["q_assigned"]]*length,
                                    "resource":[case["r"][0]]*length,
                                    "case_topic":[case["c_topic"]]*length,
                                    
                                    "initial_delay":[case["initial_delay"]]*length,
                                    
                                    "est_NPS_priority":[case["est_NPS_priority"]]*length,
                                    "est_NPS":[case["est_NPS"]]*length,
                                    "est_throughput_time":[case["est_throughputtime"]]*length,
                                    
                                    "event_no":case["j"],
                                    "activity":case["a"],
                                    
                                    "activity_start":case["t_start"],
                                    "activity_end":case["t_end"],
                                    
                                    "activity_start_dt":[datetime.datetime.strftime(i, '%m/%d/%Y %H:%M:%S') for i in case["t_start_dt"]],
                                    "activity_end_dt":[datetime.datetime.strftime(i, '%m/%d/%Y %H:%M:%S') for i in case["t_end_dt"]],
                                    
                                    "activity_start_delay":case["t_start_delay"],
                                    "activity_end_delay":case["t_end_delay"],
                                    
                                    "duration":case["t"],
                                    "duration_delayed":case["t_delayed"],
                                    
                                    "case_status":[case["status"]]*length,
                                    "F_priority_scheme":[P_scheme]*length,
                                    "F_number_of_agents":[agents]*length,
                                    "F_ceiling_value":[F_ceiling_value]*length,
                                    "F_days":[F_days]*length,
                                    "F_burn_in":[F_burn_in]*length,
                                    "F_tNPS_wtime_effect_bias":[F_tNPS_wtime_effect_bias]*length,
                                    "F_NPS_dist_bias":[F_NPS_dist_bias]*length,
                                    "seed":[seed]*length,
                                    
                                    "simulated_NPS":[NPS_simulated]*length,
                                    "simulated_NPS_priority":[NPS_priority_simulated]*length,
                                    "simulated_throughput_time":[throughput_time]*length
                                    },
                                   index=list(range(0,length)))
                
                DF_list.append(res_i)
        
        Evlog = pd.concat(DF_list, ignore_index=True)
        Evlog = Evlog.sort_values("case_id")
        
        return Evlog
    
    import datetime
    import time
    
    """
    Time series placeholder
    """
    timeseries = []
    
    """
    Generate the agents
    """
    
    from distributions.agents import generate_agents
    
    Psi = generate_agents(agents, seed)

    """
    Generate simulated activities using same seed
    """
    
    from distributions.p_vectors import generate_p_vectors
    
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
    
    Theta, i = CaseArrival(i, F_days, P_scheme, startdate, seed=seed)
    
    
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
    for d in list(range(0,F_days)):
        
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
        Case_DB.loc[:, "case_currently_arrived"] = Case_DB["arrival_q"] <= d+1
                
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
        cases_waiting_in_queue.loc[:, "queue_waiting_time"] = d - cases_waiting_in_queue["arrival_q"]
                        
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
                             "avg_current_queue_waitingtime":np.mean(cases_waiting_in_queue["queue_waiting_time"]) if len(cases_waiting_in_queue) > 0 else 0,
                             "min_current_queue_waitingtime":np.min(cases_waiting_in_queue["queue_waiting_time"]) if len(cases_waiting_in_queue) > 0 else 0,
                             "max_current_queue_waitingtime":np.max(cases_waiting_in_queue["queue_waiting_time"]) if len(cases_waiting_in_queue) > 0 else 0,
                             "median_current_queue_waitingtime":np.median(cases_waiting_in_queue["queue_waiting_time"]) if len(cases_waiting_in_queue) > 0 else 0,
                             
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
            Case_DB.loc[:, "case_currently_arrived"] = Case_DB["arrival_q"] <= z
            
            # a case is in the queue if it has currently arrived, and no agent is assigned to it
            Case_DB.loc[:, "case_queued"] = Case_DB["case_currently_arrived"] & Case_DB['agent_idx'].isnull()
            
            """ If there are currently any cases in the queue or being processed """
            queue_size = np.sum(Case_DB["case_queued"]*1)
            open_cases = len(Case_DB.loc[np.where(Case_DB['agent_idx'].notnull())])
            
            if queue_size + open_cases > 0:
                
                """ Sort case order, based on queue priority scheme """        
                Case_DB = QueueManagement(Case_DB, 
                                          P_scheme,
                                          ceiling,
                                          F_ceiling_value, 
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
    
    evlog = store_evlog(L, P_scheme, agents, filedest=filename, F_burn_in=F_burn_in, F_NPS_dist_bias=F_NPS_dist_bias, seed=seed, F_tNPS_wtime_effect_bias=F_tNPS_wtime_effect_bias)
    
    # Generate daily timeseries snapshots for this simulation run
    timeseries = pd.DataFrame(timeseries)
    
    # if there are no customers in the queue to calculate metrics from
    # NaN values will be present: therefore we impute with zeros
    timeseries = timeseries.fillna(0)
    
    # Mark the burn_in period:
    timeseries["burn_in_period"] = timeseries.day < F_burn_in #".loc[timeseries.day > burn_in-1]
    
    timeseries.to_csv("results/"+str(seed)+"/"+str(seed)+"_timeseries.csv",index=False)
    
    return evlog, Case_DB


#evlog.est_NPS_priority