# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:34:06 2022

@author: Mike



P_scheme = "FCFS"

Theta = [{'i': 20, 'q': 10.758166076206798, 'c': ['a', 'b', 'c']},
         {'i': 19, 'q': 10.3790830381034, 'c': ['a', 'b', 'c']}]

"""



def QueueManagement(Case_DB, 
                    P_scheme, 
                    ceiling, 
                    F_ceiling_value, 
                    time_interval, 
                    current_day=0,
                    F_burn_in=0,
                    seed=2021, 
                    verbose=False):
    """
    This function determines the queue order based on the chosen priority scheme.
    
    During the burn-in period, FCFS is automatically applied regardless of P_scheme
    to ensure dynamic models are trained on FCFS-prioritized data only.
    
    Parameters:
    -----------
    Case_DB : DataFrame
        The case database with all cases
    P_scheme : str
        Priority scheme to use (FCFS, SIRO, SRTF, LRTF, NPS)
        Note: Automatically overridden to FCFS during burn-in period
    ceiling : str
        Type of ceiling (NO, SLA)
    F_ceiling_value : float
        Value for the ceiling if using SLA
    time_interval : float
        Current time interval
    current_day : float
        Current simulation day (for burn-in logic)
    F_burn_in : int
        Number of burn-in days
    seed : int
        Random seed for reproducibility
    verbose : bool
        Whether to print detailed information
        
    Returns:
    --------
    Case_DB : DataFrame
        Updated case database with queue order
    """
    
    import numpy as np
    np.random.seed(seed)
    
    # Override P_scheme to FCFS during burn-in period
    original_P_scheme = P_scheme
    if current_day < F_burn_in and P_scheme != "FCFS":
        P_scheme = "FCFS"
        if verbose:
            print(f"Day {current_day:.3f}: Burn-in period - overriding {original_P_scheme} with FCFS")
    elif current_day >= F_burn_in and verbose and original_P_scheme != "FCFS":
        print(f"Day {current_day:.3f}: Post burn-in - using {original_P_scheme}")
    
    """
    Only use queued cases for sorting
    """
    # We only get the queued cases, not the ones that are started.
    queued_case_mask = (Case_DB["case_queued"] == True)
    queued_cases = Case_DB.loc[queued_case_mask].copy()
    
    # If there are cases in the queue then apply priority scheme.
    if len(queued_cases) > 0:
        
        """
        Sort cases according to priority scheme (policy)
        """
    
        # First come first served
        if P_scheme == "FCFS":
            queued_cases["queue_order"] = list(range(0,len(queued_cases)))
            
        # Service in random order
        elif P_scheme == "SIRO":
            random_order = np.random.permutation(len(queued_cases)).tolist()
            queued_cases["queue_order"] = random_order
        
        # Shortest remaining time first
        elif P_scheme == "SRTF":
            queue_order = queued_cases.sort_values("est_throughputtime")["theta_idx"].tolist()
            queued_cases["queue_order"] = list(range(0,len(queued_cases)))
            
        # Longest remaining time first
        elif P_scheme == "LRTF":
            queue_order = queued_cases.sort_values("est_throughputtime", ascending=False)["theta_idx"].tolist()
            queued_cases["queue_order"] = list(range(0,len(queued_cases)))
            
        # NPS-routing: Sort on expected NPS outcome.
        elif P_scheme == "NPS": 
            queue_order = queued_cases.sort_values("est_NPS_priority", ascending=False)["theta_idx"].tolist()
            queued_cases["queue_order"] = list(range(0,len(queued_cases)))
          
            
        """
        Implement the SLA ceiling mechanism (activate when time has passed, and set priority to 0)
        """
        if ceiling == "SLA":
            # Time since arrival at current moment
            queued_cases["time_since_arrival"] = time_interval - queued_cases["arrival_q"]
            
            # Cases with waiting time longer than ceiling_value days (=oldest cases)
            # (note: lower queue_order means higher priority)
            queued_cases.loc[queued_cases["time_since_arrival"] >= F_ceiling_value,"queue_order"] = -1
        
        
        # Update the queue, SLA cases first. Then in order of arrival
        if ceiling == "SLA":
            queued_cases = queued_cases.sort_values(["queue_order","arrival_q"])
        else:
            queued_cases = queued_cases.sort_values("queue_order")
            
        # copy back data to the original dataset
        Case_DB.loc[queued_case_mask,"queue_order"] = queued_cases["queue_order"].values
    
    return Case_DB

