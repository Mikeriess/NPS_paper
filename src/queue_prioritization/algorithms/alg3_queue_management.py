# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:34:06 2022

@author: Mike



P_scheme = "FCFS"

Theta = [{'i': 20, 'q': 10.758166076206798, 'c': ['a', 'b', 'c']},
         {'i': 19, 'q': 10.3790830381034, 'c': ['a', 'b', 'c']}]

"""



def QueueManagement(Case_DB, P_scheme, ceiling, ceiling_value, time_interval, seed):
    """
    Parameters
    ----------
    Theta : TYPE
        DESCRIPTION.
    P_scheme : TYPE
        DESCRIPTION.

    Returns
    -------
    Theta_ordered

    """
    import numpy as np
    np.random.seed(seed)
    import pandas as pd
    import random
    
    #sort case db by arrival time
    #Case_DB = Case_DB.sort_values("arrival_q",ascending=True)
    #Case_DB.index = list(range(0,len(Case_DB)))    
    #Case_DB["queue_order"] = list(range(0,len(Case_DB)))


    """ Only cases that arrived up until time z + 15 is visible """



    """ FCFS - First-come first served queue management """
    if P_scheme == "FCFS":    
        
        """ Sort cases by their arrival time q"""
        #Theta_ordered = sorted(Theta.copy(), key=lambda d: d['q']) 
        
        #sort case db by arrival time
        Case_DB = Case_DB.sort_values("arrival_q",ascending=True)
        
        #change the index
        Case_DB.index = list(range(0,len(Case_DB)))
        
        #set the queue order (including cases that are not in the queue)
        Case_DB["queue_order"] = list(range(0,len(Case_DB)))
    

    # """ SIRO - Service In Random Order """
    # if P_scheme == "SIRO":    
                
    #     """
    #     Make random permutations to the queue
    #     """
        
    #     #Shuffle the dataframe
    #     Case_DB = Case_DB.sample(frac = 1)
        
    #     #Update the indexes
    #     Case_DB.index = list(range(0,len(Case_DB)))
    #     Case_DB["queue_order"] = list(range(0,len(Case_DB)))
                
        
    """ SRTF - Shortest Remaining Time First """
    if P_scheme == "SRTF":    
                
        #sort case db by the predicted throughput time
        Case_DB = Case_DB.sort_values("est_throughputtime", ascending=True)
        
        #change the index
        Case_DB.index = list(range(0,len(Case_DB)))
        
        #set the queue order (including cases that are not in the queue)
        Case_DB["queue_order"] = list(range(0,len(Case_DB)))
        
    
    
    """ LRTF - Longest Remaining Time First """
    if P_scheme == "LRTF":    
                
        #sort case db by the predicted throughput time
        Case_DB = Case_DB.sort_values("est_throughputtime", ascending=False)
        
        #change the index
        Case_DB.index = list(range(0,len(Case_DB)))
        
        #set the queue order (including cases that are not in the queue)
        Case_DB["queue_order"] = list(range(0,len(Case_DB)))
        
        
    """ NPS-based queue management """
    if P_scheme == "NPS":
        
        """ If there is a tie, the one that arrived first will get first served """
        
        #sort case db by arrival time
        Case_DB = Case_DB.sort_values("arrival_q",ascending=True)
        Case_DB.index = list(range(0,len(Case_DB)))
        
        #sort case db by NPS_priority
        Case_DB = Case_DB.sort_values("est_NPS_priority",ascending=True)
        Case_DB.index = list(range(0,len(Case_DB)))
        
        #set the queue order (including cases that are not in the queue)
        Case_DB["queue_order"] = list(range(0,len(Case_DB)))
        
    # """ NPS-based queue management """
    # if P_scheme == "NPS-SLA":
        
    #     """ If there is a tie, the one that arrived first will get first served """
        
    #     #sort case db by arrival time
    #     Case_DB = Case_DB.sort_values("arrival_q",ascending=True)
    #     Case_DB.index = list(range(0,len(Case_DB)))
        
    #     #sort case db by NPS_priority
    #     Case_DB = Case_DB.sort_values("est_NPS_priority",ascending=True)
    #     Case_DB.index = list(range(0,len(Case_DB)))
        
    #     #set the queue order (including cases that are not in the queue)
    #     Case_DB["queue_order"] = list(range(0,len(Case_DB)))
    
    """
    Post-prioritization: Service level agreement
    """    
    if ceiling == "SLA":
        """
        get queue waiting time
        """        
        # Calculate waiting time: if negative, there is no waiting time yet
        Case_DB["queue_waiting_time"] = time_interval - Case_DB["arrival_q"]
        
        """
        re-prioritize the queue, given current waiting time
        """
        
        #subset the priority customers that are close to having SLA-violations
        priority_customers = Case_DB.loc[Case_DB.queue_waiting_time >= ceiling_value]
        
        #sort them by the time they have waited
        priority_customers = priority_customers.sort_values("queue_waiting_time", ascending=False)
        
        #get the rest of the customers, and keep their order
        remaining_customers = Case_DB.loc[Case_DB.queue_waiting_time < ceiling_value]
        
        #combine the two to a new list
        New_Case_DB = pd.concat([priority_customers, remaining_customers],axis=0)
        New_Case_DB.index = list(range(0,len(New_Case_DB)))
        
        #set the queue order (including cases that are not in the queue)
        New_Case_DB["queue_order"] = list(range(0,len(New_Case_DB)))
        
        #overwrite the old case DB
        Case_DB = New_Case_DB.copy()
    
    # id variable for later use in case assignment
    Case_DB["casedb_idx"] = list(range(0,len(Case_DB)))
    
    return Case_DB

