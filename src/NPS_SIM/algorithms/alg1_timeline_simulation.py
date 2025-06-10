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
                   verbose = False,
                   results_dir="results",
                   F_throughput_model: str = "Static",
                   F_throughput_model_penalty: float = 0.1,
                   F_nps_model: str = "Static",
                   F_nps_model_penalty: float = 0.1,
                   F_uniform_duration_mode: str = "DISABLED",
                   F_uniform_duration_minutes: int = 180
                   ):
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
    results_dir : str
        Directory where results should be stored
    F_throughput_model : str
        Type of throughput model to use/train ("Static", "Lasso", "Gamma_GLM").
    F_throughput_model_penalty : float
        Penalty (alpha) parameter for the dynamic throughput model (Lasso or Gamma_GLM).
    F_nps_model : str
        Type of NPS model to use/train ("Static", "Lasso", "Gamma_GLM").
    F_nps_model_penalty : float
        Penalty (alpha) parameter for the dynamic NPS model (Lasso or Gamma_GLM).
    F_uniform_duration_mode : str
        Enable uniform duration mode ("ENABLED" or "DISABLED"). When enabled, all cases take exactly the same duration.
    F_uniform_duration_minutes : int
        Duration in minutes for each case when uniform duration mode is enabled (default 180).
        
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
    import os
    from pathlib import Path
    
    #from simulation.helper_functions import store_evlog, sim_generalized_lognormal
    from algorithms.alg2_case_arrival import CaseArrival
    from algorithms.alg3_queue_management import QueueManagement
    from algorithms.alg4_case_assignment import CaseAssignment
    from algorithms.alg5_case_activities import CaseActivities
    
    # Define a local store_evlog function instead of importing it
    def store_evlog(L, P_scheme, agents, filedest, F_burn_in, F_NPS_dist_bias, seed, F_tNPS_wtime_effect_bias, F_days, total_days, F_uniform_duration_mode, F_uniform_duration_minutes):
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
            case["burn_in"] = case["q"] < F_burn_in
                        
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
                
                # Store simulated NPS back in the case dictionary for metrics calculation
                case["simulated_NPS"] = NPS_simulated
                case["simulated_NPS_priority"] = NPS_priority_simulated
                case["simulated_throughput_time"] = throughput_time
                
                # Prepare base dataframe data
                base_data = {"case_id":[case["i"]]*length,
                           
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
                           "effective_priority_scheme":["FCFS" if case["burn_in"] else P_scheme]*length,
                           "F_number_of_agents":[agents]*length,
                           "F_ceiling_value":[F_ceiling_value]*length,
                           "F_days":[F_days]*length,
                           "F_total_days":[total_days]*length,
                           "F_burn_in":[F_burn_in]*length,
                           "F_tNPS_wtime_effect_bias":[F_tNPS_wtime_effect_bias]*length,
                           "F_NPS_dist_bias":[F_NPS_dist_bias]*length,
                           "F_uniform_duration_mode":[F_uniform_duration_mode]*length,  
                           "F_uniform_duration_minutes":[F_uniform_duration_minutes]*length,
                           "seed":[seed]*length,
                           
                           "simulated_NPS":[NPS_simulated]*length,
                           "simulated_NPS_priority":[NPS_priority_simulated]*length,
                           "simulated_throughput_time":[throughput_time]*length
                           }
                
                # Add queue state features if they exist in the case
                queue_state_features = {k: v for k, v in case.items() if k.startswith('qs_')}
                if queue_state_features:
                    for feature_name, feature_value in queue_state_features.items():
                        base_data[feature_name] = [feature_value]*length
                
                # Generate a dataframe
                res_i = pd.DataFrame(base_data, index=list(range(0,length)))
                
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
    Calculate total simulation days
    """
    total_days = F_days + F_burn_in
    
    """
    Simulate case arrivals throughout the full period
    """
    i = 0
    counter = 0
    
    # Note: CaseArrival generates all cases upfront for the entire simulation period (burn-in + main)
    # Dynamic model usage will be handled during queue management and assignment phases
    Theta, i = CaseArrival(i, total_days, P_scheme, startdate, seed=seed)
    
    
    """
    Generate case database: All cases throughout the simulation period
    """
    # Initialize base Case_DB data
    case_db_data = {'theta_idx':list(range(0,len(Theta))), 
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
                    }
    
    # Add queue state feature columns (initialized with NaN for cases that haven't arrived yet)
    queue_feature_names = [
        'qs_agents_available', 'qs_agents_busy', 'qs_agent_utilization',
        'qs_queue_length', 'qs_cases_in_process', 'qs_total_active_cases', 'qs_queue_wait_time_current',
        'qs_recent_completion_rate', 'qs_recent_avg_throughput_time', 'qs_recent_arrival_rate', 'qs_workload_intensity',
        'qs_time_since_last_completion', 'qs_cases_arrived_today'
    ]
    
    for feature_name in queue_feature_names:
        case_db_data[feature_name] = [float('nan')] * len(Theta)
    
    Case_DB = pd.DataFrame(case_db_data)
    
    
    """ For each day in the simulation period """
    # Initialize dynamic model variables to track if training was performed
    dynamic_model_info = None
    dynamic_nps_model_info = None
    
    # Print burn-in information at start
    if F_burn_in > 0:
        print(f"Starting simulation with {total_days} days total ({F_burn_in} burn-in + {F_days} main)")
        print(f"Burn-in period (days 0-{F_burn_in-1}): Using FCFS prioritization for all cases")
        print(f"Main period (days {F_burn_in}-{total_days-1}): Using {P_scheme} prioritization")
    else:
        print(f"Starting simulation with {F_days} days total (no burn-in)")
        print(f"All periods: Using {P_scheme} prioritization")
    
    for d in list(range(0, total_days)):
        
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
        if d == total_days - 1:  # Last day of simulation
            end_time = time.time()
            Time_sec = end_time - start_time
            print("Duration in minutes:",Time_sec/60)
            
        # Train dynamic models at the end of burn-in period
        if d == F_burn_in and F_burn_in > 0:
            print(f"End of burn-in period (day {d}). All burn-in cases were prioritized using FCFS.")
            if F_throughput_model != "Static":
                print(f"Training dynamic throughput model ({F_throughput_model}) with penalty {F_throughput_model_penalty}...")
            if F_nps_model != "Static":
                print(f"Training dynamic NPS model ({F_nps_model}) with penalty {F_nps_model_penalty}...")
            print(f"Starting main period (day {d}) with priority scheme: {P_scheme}")
            
            # Ensure the run-specific directory exists for saving the models
            run_dir = Path(results_dir) / str(seed)
            run_dir.mkdir(parents=True, exist_ok=True)
            
            # Mark burn-in status on cases in L before training
            for case in L:
                case["burn_in"] = case["q"] < F_burn_in
            
            # Train dynamic throughput model
            try:
                from models.dynamic_throughput import train_model_on_burn_in
                
                dynamic_model_info = train_model_on_burn_in(
                    case_list=L, 
                    run_dir=run_dir,
                    throughput_model_type=F_throughput_model,
                    model_penalty_alpha=F_throughput_model_penalty
                )
                
                if dynamic_model_info and dynamic_model_info.get("training_successful", False):
                    print(f"Dynamic throughput model training successful. {dynamic_model_info['n_training_samples']} samples used.")
                    print(f"Throughput model saved to: {run_dir / 'models' / 'dynamic_throughput_model.json'}")
                    print(f"Dynamic throughput model trained and ready. Predictions will be updated in real-time as cases arrive.")
                else:
                    print("Dynamic throughput model training failed. Will continue using static model.")
                    dynamic_model_info = None
                    
            except Exception as e:
                print(f"Error during dynamic throughput model training: {e}")
                print("Will continue using static throughput model.")
                dynamic_model_info = None
            
            # Train dynamic NPS model
            try:
                from models.dynamic_nps import train_nps_model_on_burn_in
                
                dynamic_nps_model_info = train_nps_model_on_burn_in(
                    case_list=L, 
                    run_dir=run_dir,
                    nps_model_type=F_nps_model,
                    model_penalty_alpha=F_nps_model_penalty
                )
                
                if dynamic_nps_model_info and dynamic_nps_model_info.get("training_successful", False):
                    print(f"Dynamic NPS model training successful. {dynamic_nps_model_info['n_training_samples']} samples used.")
                    print(f"NPS model saved to: {run_dir / 'models' / 'dynamic_nps_model.json'}")
                    print(f"Dynamic NPS model trained and ready. Predictions will be updated in real-time as cases arrive.")
                else:
                    print("Dynamic NPS model training failed. Will continue using static model.")
                    dynamic_nps_model_info = None
                    
            except Exception as e:
                print(f"Error during dynamic NPS model training: {e}")
                print("Will continue using static NPS model.")
                dynamic_nps_model_info = None
        
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
                    
        # Calculate queue waiting times directly without modifying a DataFrame slice
        queue_mask = Case_DB["case_queued"] == True
        queue_waiting_times = d - Case_DB.loc[queue_mask, "arrival_q"] if queue_mask.any() else pd.Series(dtype=float)
                        
        # Calculate queue state features for the daily snapshot
        from models.queue_state_features import calculate_queue_state_features
        daily_queue_features = calculate_queue_state_features(Case_DB, Psi, d+1, lookback_window=1.0)
        
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
                             "avg_current_queue_waitingtime":np.mean(queue_waiting_times) if len(queue_waiting_times) > 0 else 0,
                             "min_current_queue_waitingtime":np.min(queue_waiting_times) if len(queue_waiting_times) > 0 else 0,
                             "max_current_queue_waitingtime":np.max(queue_waiting_times) if len(queue_waiting_times) > 0 else 0,
                             "median_current_queue_waitingtime":np.median(queue_waiting_times) if len(queue_waiting_times) > 0 else 0,
                             
                             #cases being processed
                             # "avg_total_waitingtime":np.mean(cases_waiting["waiting_time"]),
                             # "min_total_waitingtime":np.min(cases_waiting["waiting_time"]),
                             # "max_total_waitingtime":np.max(cases_waiting["waiting_time"]),
                             # "median_total_waitingtime":np.median(cases_waiting["waiting_time"]),
                             }
        
        # Add daily queue state features to the morning snapshot
        for feature_name, feature_value in daily_queue_features.items():
            morning_snapshot[f"daily_{feature_name}"] = feature_value
                
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
            # Store previous arrival state to detect newly arrived cases
            previously_arrived = Case_DB["case_currently_arrived"].copy()
            
            # a case has arrived if its arrival time is before the end of the current window
            Case_DB.loc[:, "case_currently_arrived"] = Case_DB["arrival_q"] <= z
            
            # Identify newly arrived cases in this interval
            newly_arrived_mask = (Case_DB["case_currently_arrived"]) & (~previously_arrived)
            
            # Calculate queue state features for newly arrived cases
            if newly_arrived_mask.any():
                from models.queue_state_features import calculate_queue_state_features, extend_case_with_queue_features
                
                queue_features = calculate_queue_state_features(Case_DB, Psi, z)
                
                # Update Theta with queue state features for newly arrived cases
                newly_arrived_indices = Case_DB.loc[newly_arrived_mask, "theta_idx"].values
                for theta_idx in newly_arrived_indices:
                    Theta[theta_idx] = extend_case_with_queue_features(Theta[theta_idx], queue_features)
                
                # Also update Case_DB with queue state features for newly arrived cases
                for feature_name, feature_value in queue_features.items():
                    Case_DB.loc[newly_arrived_mask, f'qs_{feature_name}'] = feature_value
                
                # Real-time dynamic prediction updates for newly arrived cases (if after burn-in)
                if d >= F_burn_in:
                    # Dynamic throughput prediction
                    if dynamic_model_info and dynamic_model_info.get("training_successful", False):
                        from models.dynamic_throughput import predict_TT_dynamic
                        
                        for theta_idx in newly_arrived_indices:
                            try:
                                # Get updated prediction using dynamic model with queue state features
                                updated_case = predict_TT_dynamic(Theta[theta_idx], dynamic_model_info)
                                
                                # Update both Theta and Case_DB with new prediction
                                Theta[theta_idx]["est_throughputtime"] = updated_case["est_throughputtime"]
                                Case_DB.loc[theta_idx, "est_throughputtime"] = updated_case["est_throughputtime"]
                                
                            except Exception as e:
                                # If dynamic prediction fails, keep the original static prediction
                                pass
                    
                    # Dynamic NPS prediction (after throughput prediction is updated)
                    if dynamic_nps_model_info and dynamic_nps_model_info.get("training_successful", False):
                        from models.dynamic_nps import predict_NPS_dynamic
                        
                        for theta_idx in newly_arrived_indices:
                            try:
                                # Get updated NPS prediction using dynamic model
                                updated_case = predict_NPS_dynamic(Theta[theta_idx], dynamic_nps_model_info)
                                
                                # Update both Theta and Case_DB with new NPS prediction
                                Theta[theta_idx]["est_NPS"] = updated_case["est_NPS"]
                                Theta[theta_idx]["est_NPS_priority"] = updated_case["est_NPS_priority"]
                                Case_DB.loc[theta_idx, "est_NPS"] = updated_case["est_NPS"]
                                Case_DB.loc[theta_idx, "est_NPS_priority"] = updated_case["est_NPS_priority"]
                                
                            except Exception as e:
                                # If dynamic NPS prediction fails, keep the original static prediction
                                pass
            
            # a case is in the queue if it has currently arrived, and no agent is assigned to it
            Case_DB.loc[:, "case_queued"] = Case_DB["case_currently_arrived"] & Case_DB['agent_idx'].isnull()
            
            """ If there are currently any cases in the queue or being processed """
            queue_size = np.sum(Case_DB["case_queued"]*1)
            open_cases = len(Case_DB.loc[np.where(Case_DB['agent_idx'].notnull())])
            
            if queue_size + open_cases > 0:
                
                """ Sort case order, based on queue priority scheme """        
                # Determine effective priority scheme: use FCFS during burn-in, actual P_scheme after
                effective_P_scheme = "FCFS" if d < F_burn_in else P_scheme
                
                Case_DB = QueueManagement(Case_DB, 
                                          effective_P_scheme,
                                          ceiling,
                                          F_ceiling_value, 
                                          time_interval = z,
                                          current_day = d,
                                          F_burn_in = F_burn_in,
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
                                                        counter=counter,
                                                        F_uniform_duration_mode=F_uniform_duration_mode,
                                                        F_uniform_duration_minutes=F_uniform_duration_minutes)
                
            
    
    """
    Save an event-log
    """
    
    arrived_cases = len(Case_DB)
    
    evlog = store_evlog(L, P_scheme, agents, filedest=filename, F_burn_in=F_burn_in, F_NPS_dist_bias=F_NPS_dist_bias, seed=seed, F_tNPS_wtime_effect_bias=F_tNPS_wtime_effect_bias, F_days=F_days, total_days=total_days, F_uniform_duration_mode=F_uniform_duration_mode, F_uniform_duration_minutes=F_uniform_duration_minutes)
    
    # Calculate main period performance metrics if dynamic models were used
    main_period_metrics = {"mae_main": float('nan'), "mse_main": float('nan'), "n_main_cases": 0}
    main_period_nps_metrics = {"mae_main_nps": float('nan'), "mse_main_nps": float('nan'), "n_main_cases_nps": 0}
    
    # Throughput model metrics
    if dynamic_model_info and dynamic_model_info.get("training_successful", False):
        try:
            from models.dynamic_throughput import calculate_main_period_metrics
            mae_main, mse_main, n_main = calculate_main_period_metrics(L, dynamic_model_info, F_burn_in)
            main_period_metrics = {
                "mae_main": mae_main, 
                "mse_main": mse_main, 
                "n_main_cases": n_main
            }
            print(f"Main period throughput model performance: MAE={mae_main:.4f}, MSE={mse_main:.4f}, N={n_main}")
        except Exception as e:
            print(f"Error calculating main period throughput metrics: {e}")
    
    # NPS model metrics
    if dynamic_nps_model_info and dynamic_nps_model_info.get("training_successful", False):
        try:
            from models.dynamic_nps import calculate_main_period_metrics_nps
            mae_main_nps, mse_main_nps, n_main_nps = calculate_main_period_metrics_nps(L, dynamic_nps_model_info, F_burn_in)
            main_period_nps_metrics = {
                "mae_main_nps": mae_main_nps, 
                "mse_main_nps": mse_main_nps, 
                "n_main_cases_nps": n_main_nps
            }
            print(f"Main period NPS model performance: MAE={mae_main_nps:.4f}, MSE={mse_main_nps:.4f}, N={n_main_nps}")
        except Exception as e:
            print(f"Error calculating main period NPS metrics: {e}")
    
    # Generate daily timeseries snapshots for this simulation run
    timeseries = pd.DataFrame(timeseries)
    
    # if there are no customers in the queue to calculate metrics from
    # NaN values will be present: therefore we impute with zeros
    timeseries = timeseries.fillna(0)
    
    # Mark the burn_in period:
    timeseries["burn_in_period"] = timeseries.day < F_burn_in
    
    # Ensure the main results directory exists
    results_path = Path(results_dir)
    if not results_path.exists():
        os.makedirs(results_path, exist_ok=True)
    
    # Ensure the run-specific directory exists
    run_dir = results_path / str(seed)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save timeseries to the specified results directory
    timeseries.to_csv(run_dir / f"{seed}_timeseries.csv", index=False)
    
    # Combine dynamic model training and main period metrics
    model_metrics = {}
    
    # Throughput model metrics
    if dynamic_model_info and dynamic_model_info.get("training_successful", False):
        model_metrics.update({
            "mae_burnin": dynamic_model_info.get("mae_burnin", float('nan')),
            "mse_burnin": dynamic_model_info.get("mse_burnin", float('nan')),
            "n_burnin_training_samples": dynamic_model_info.get("n_training_samples", 0)
        })
    else:
        model_metrics.update({
            "mae_burnin": float('nan'),
            "mse_burnin": float('nan'), 
            "n_burnin_training_samples": 0
        })
    
    # NPS model metrics
    if dynamic_nps_model_info and dynamic_nps_model_info.get("training_successful", False):
        model_metrics.update({
            "mae_burnin_nps": dynamic_nps_model_info.get("mae_train", float('nan')),
            "mse_burnin_nps": dynamic_nps_model_info.get("mse_train", float('nan')),
            "n_burnin_training_samples_nps": dynamic_nps_model_info.get("n_training_samples", 0)
        })
    else:
        model_metrics.update({
            "mae_burnin_nps": float('nan'),
            "mse_burnin_nps": float('nan'), 
            "n_burnin_training_samples_nps": 0
        })
    
    model_metrics.update(main_period_metrics)
    model_metrics.update(main_period_nps_metrics)
    
    return evlog, Case_DB, model_metrics


#evlog.est_NPS_priority