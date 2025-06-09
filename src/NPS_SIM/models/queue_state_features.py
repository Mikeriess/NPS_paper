"""
Queue State Feature Calculation for Dynamic Throughput Prediction

This module calculates observable queue state features at the time
when new cases arrive, for use in dynamic throughput prediction models.
"""

import numpy as np
import pandas as pd

def calculate_queue_state_features(Case_DB, Psi, current_time, lookback_window=1.0):
    """
    Calculate queue state features at case arrival time.
    
    Parameters:
    -----------
    Case_DB : pd.DataFrame
        Current case database with arrival times, status, etc.
    Psi : list
        Agent pool with current status information  
    current_time : float
        Current simulation time (in days)
    lookback_window : float
        Time window in days for calculating recent metrics (default: 1.0 day)
    
    Returns:
    --------
    dict : Dictionary of queue state features
    """
    
    # Agent Availability Features (3 features)
    n_agents = len(Psi)
    busy_agents = sum(1 for agent in Psi if agent.get("status") == "assigned to case")
    agents_available = n_agents - busy_agents
    agent_utilization = busy_agents / n_agents if n_agents > 0 else 0.0
    
    # Queue State Features (4 features)
    queue_length = (Case_DB["case_queued"] == True).sum()
    cases_in_process = (Case_DB['agent_idx'].notnull()).sum()
    total_active_cases = queue_length + cases_in_process
    
    # Current queue waiting time for cases in queue
    if queue_length > 0:
        queue_mask = Case_DB["case_queued"] == True
        queue_wait_times = current_time - Case_DB.loc[queue_mask, "arrival_q"]
        queue_wait_time_current = queue_wait_times.mean()
    else:
        queue_wait_time_current = 0.0
    
    # System Throughput Features (4 features) - Calculate from closed cases in lookback window
    lookback_start = current_time - lookback_window
    
    # Cases that were closed in the lookback window
    closed_cases = Case_DB[Case_DB["status"] == "closed"]
    
    if len(closed_cases) > 0:
        # Get cases with some closure information - this is a simplification
        # In a real system, we'd need actual closure times from the event log
        # For now, we'll estimate based on available data
        
        recent_arrivals = Case_DB[Case_DB["arrival_q"] >= lookback_start]
        recent_arrival_rate = len(recent_arrivals) / lookback_window
        
        # Estimate completion rate from closed cases
        recent_completion_rate = len(closed_cases) / current_time if current_time > 0 else 0.0
        
        # Estimate average throughput time from existing predictions
        recent_avg_throughput_time = Case_DB["est_throughputtime"].mean()
        
        # Workload intensity (arrival rate / service rate approximation)
        service_rate_estimate = recent_completion_rate if recent_completion_rate > 0 else 0.001
        workload_intensity = recent_arrival_rate / service_rate_estimate
    else:
        recent_arrival_rate = 0.0
        recent_completion_rate = 0.0
        recent_avg_throughput_time = 0.0
        workload_intensity = 0.0
    
    # Temporal Context Features (2 features)
    if len(closed_cases) > 0:
        # This is a simplification - ideally we'd track actual completion times
        time_since_last_completion = 0.1  # Placeholder - would need event log data
    else:
        time_since_last_completion = current_time
    
    # Cases that arrived today (same integer day)
    current_day = int(current_time)
    cases_arrived_today = ((Case_DB["arrival_q"] >= current_day) & 
                          (Case_DB["arrival_q"] < current_day + 1)).sum()
    
    # Compile all features
    features = {
        # Agent Availability (3)
        'agents_available': agents_available,
        'agents_busy': busy_agents, 
        'agent_utilization': agent_utilization,
        
        # Queue State (4)
        'queue_length': queue_length,
        'cases_in_process': cases_in_process,
        'total_active_cases': total_active_cases,
        'queue_wait_time_current': queue_wait_time_current,
        
        # System Throughput (4)
        'recent_completion_rate': recent_completion_rate,
        'recent_avg_throughput_time': recent_avg_throughput_time,
        'recent_arrival_rate': recent_arrival_rate,
        'workload_intensity': workload_intensity,
        
        # Temporal Context (2)
        'time_since_last_completion': time_since_last_completion,
        'cases_arrived_today': cases_arrived_today
    }
    
    return features


def extend_case_with_queue_features(case_dict, queue_features):
    """
    Add queue state features to a case dictionary.
    
    Parameters:
    -----------
    case_dict : dict
        Case dictionary from Theta
    queue_features : dict  
        Queue state features from calculate_queue_state_features()
        
    Returns:
    --------
    dict : Updated case dictionary with queue features
    """
    
    # Create a copy to avoid modifying the original
    extended_case = case_dict.copy()
    
    # Add all queue features with 'qs_' prefix to avoid naming conflicts
    for feature_name, feature_value in queue_features.items():
        extended_case[f'qs_{feature_name}'] = feature_value
    
    return extended_case 