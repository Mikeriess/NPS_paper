# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 11:43:41 2022

@author: Mike
"""





def store_evlog(L, P_scheme, agents, filedest, burn_in, NPS_bias, seed):
    import pandas as pd
    import numpy as np
    import datetime
    #print("generating eventlog...")
    
    from distributions.tNPS import simulate_NPS

    DF_list = []
    
    #step counter for random values seed
    step = seed
    
    
    for case in L:
        # update step counter used for seed, each case gets its unique seed
        step = step +1
        
        # Generate lists of equal length to be appended
    
        #trace length 
        length = len(case["a"])
        
        #only store closed cases
        #if case["status"] == "closed":
            
        # Mark the burn_in period cases
        case["burn_in"] = case["q"] < burn_in-1
                    
        if length > 0:
            # Calculate throughput time (incl. queue waiting time)
            throughput_time = np.max(case["t_end"])-case["q"]
            
            # Simulate NPS score after case closure
            NPS_simulated, NPS_priority_simulated = simulate_NPS(case_topicidx=case["c_topic"], 
                                                                 y=throughput_time, 
                                                                 seed=step, 
                                                                 bias=NPS_bias)#[0]
            
            # Generate a dataframe
            res_i = pd.DataFrame({"case_id":[case["i"]]*length,
                                  
                                  "case_arrival":[case["q"]]*length,
                                  "burn_in_period":[case["burn_in"]]*length,
                                  
                                  #"case_arrival_offset":[case["q_offset"]]*length,
                                  
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
                                  
                                  #"activity_start":case["t_start_dt"].strftime('%m/%d/%Y %H:%M:%S'),
                                  #"activity_end":case["t_end_dt"].strftime('%m/%d/%Y %H:%M:%S'),
                                  
                                  "activity_start_dt":[datetime.datetime.strftime(i, '%m/%d/%Y %H:%M:%S') for i in case["t_start_dt"]],
                                  "activity_end_dt":[datetime.datetime.strftime(i, '%m/%d/%Y %H:%M:%S') for i in case["t_end_dt"]],
                                  
                                  "activity_start_delay":case["t_start_delay"],
                                  "activity_end_delay":case["t_end_delay"],
                                  
                                  "duration":case["t"],
                                  "duration_delayed":case["t_delayed"],
                                  
                                  #"delay_type":[case["delay_type"][0]]*length,
                                  "case_status":[case["status"]]*length,
                                  "F_priority_scheme":[P_scheme]*length,
                                  "F_number_of_agents":[agents]*length,
                                  "seed":[seed]*length,
                                  
                                  "simulated_NPS":[NPS_simulated]*length,
                                  "simulated_NPS_priority":[NPS_priority_simulated]*length,
                                  "simulated_throughput_time":[throughput_time]*length #end of last event and arrival time
                                  },
                                 index=list(range(0,length)))
            
            DF_list.append(res_i)
     
    Evlog = pd.concat(DF_list, ignore_index=True)
    Evlog = Evlog.sort_values("case_id")
    
    return Evlog 


