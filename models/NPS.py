def predict_NPS_Notopic(sigma):
     
    import numpy as np
    #import pandas as pd
    #from mpmath import gamma

    # update case with estimated throughput time
    y = sigma["est_throughputtime"]
    
    
    # convert from days to minutes
    y = y*(24*60) 
    
    """
    ################################################
    NPS score prediction 
    ################################################
    """
    
    # get features
    log_throughputtime = np.log(1+y)

    # model params
    intercept = 10.221063
    #scale = 1.3378604
    
    #coefficients
    betas = [-0.094867]
    
    #inputs
    X = [log_throughputtime]    
   
    #prediction equation
    NPS = (intercept + np.dot(X, betas))-1
    
    # update case with estimated throughput time
    sigma["est_NPS"] = NPS
    
    # update weighted NPS_priority:
    NPS_priority = np.abs(NPS-7.5)
    sigma["est_NPS_priority"] = NPS_priority
    
    return sigma



def predict_NPS(sigma):
    import numpy as np
    """
    Predict NPS from TT and case information
    """

    # update case with estimated throughput time
    y = sigma["est_throughputtime"]
    
    # convert from days to minutes
    y = y*(24*60) 
    
    
    """
    ################################################
    NPS score prediction 
    ################################################
    """
    
    # get features
    log_throughputtime = np.log(1+y)

    # model params
    intercept = 2.300587
    gammascale = 1.300057
    
    #normalscale = 2.3027043599
    
    #coefficients
    betas = [-0.0098232, #log_throughputtime
             -0.12910, #d2
             0.100756, #g1
             0.085297, #j1
             -0.042748, #q3
             -0.031668, #r2
             0.0, #w1
             0.0, #w2
             0.09665, #z2
             -0.00047, #z3
             0.0 #z4
             ]
    
    #inputs
    X = [log_throughputtime,
             d_2,
            g_1,
            j_1,
            q_3,
            r_2,
            w_1,
            w_2,
            z_2,
            z_3,
            z_4]
                
    #prediction equation
    NPS = (np.exp(intercept + np.dot(X,betas))/gammascale)-1
    
    # update case with estimated throughput time
    sigma["est_NPS"] = NPS
    
    # update weighted NPS_priority:
    NPS_priority = np.abs(NPS-7.5)
    sigma["est_NPS_priority"] = NPS_priority
    
    return sigma