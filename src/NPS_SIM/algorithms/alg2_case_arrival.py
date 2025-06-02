# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 11:06:14 2022

@author: Mike
"""


def CaseArrival(i, D, P_scheme, startdate, seed):
    import numpy as np
    import pandas as pd
    
    #set seed
    np.random.seed(seed)
       
    import datetime
    
    date_and_time = startdate #datetime.datetime(2018, 7, 1, 0, 0, 0)
    print("simulation period start:",date_and_time)
    
    q = 0
    z = 0

    M = []
    
    """ simulate all case topics beforehand """
    
    topics = list(range(0,10))
    
    P_topics = [0.318670,
                0.243781,
                0.138256,
                0.066510,
                0.064415,
                0.063891,
                0.039277,
                0.037968,
                0.014925,
                0.012307]
    
    sim_topics = np.random.choice(topics, size=100000, replace=True, p=P_topics)
    
    uniform_values = np.random.uniform(0,1, size=100000)
    
    """ Simulate todays case arrivals """
    
    for step in range(0,10000000000000):
        
        """
        Determine current place in time
        """
        
        time_change = datetime.timedelta(days=q)
        date_and_time = date_and_time + time_change
        #print(date_and_time)
    
        """
        ################################################
        Inter-arrival time prediction 
        ################################################
        """
            
        # Inputs for case arrival model
        #hour = date_and_time.hour
        weekday = date_and_time.weekday() #np.mod(z,7)
        month = date_and_time.month
        day = date_and_time.day
        year = date_and_time.year
                
        
        X = [year,
             month,
             day,
             weekday]
        
        betas = [-0.3589,
                 -0.0881,
                  0.0078,
                  0.2616]
        
        c = 726.6267
        
        #simulated residuals
        residual = -np.log(1-uniform_values[step])
        
        #Simulation equation: exponential regression
        q = residual * np.exp(c + np.dot(X,betas))
        
        #prediction expression is in hours, convert back to days:            
        q = q/24
        
              
        """ Simulate a new case arrival at time q in day d, and append it to the set of cases M """
        sigma = {"i":i,
                 "q":z+q, 
                 "q_offset":q,
                 "q_dt":date_and_time + datetime.timedelta(days=q),
                 "q_assigned":0,
                 "r":[],
                 "c_topic":sim_topics[step],
                 'j':[],
                 'a':[],
                 't':[],
                 't_delayed':[],
                 't_start':[],
                 't_end':[],
                 't_start_dt':[],
                 't_end_dt':[],
                 't_start_delay':[],
                 't_end_delay':[],
                 'initial_delay':0,
                 "status":"queue",
                 
                 "est_throughputtime":0,
                 "est_NPS":0,
                 "est_NPS_priority":0}
        
        
        """predict the expected RT and NPS for the given case"""
                        
        #predict rt and nps
        from models.NPS import predict_NPS_Notopic, predict_NPS
        from models.throughput import predict_TT_Notopic, predict_TT
        
        # Predict throughput time
        #sigma = predict_TT_Notopic(sigma)
        sigma = predict_TT(sigma)
        
        # Predict NPS
        #sigma = predict_NPS_Notopic(sigma)
        sigma = predict_NPS(sigma)
        
        # update time and counter variables
        if sigma["q"] <= D+1:
            #print(i,sigma["q"])
            M.append(sigma)
            z = z + q            
            i = i + 1            
        else:
            break
    
    return M.copy(), i




    