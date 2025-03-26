
def simulate_NPS(case_topicidx, y, seed, bias=0):
    
    import numpy as np
    import pandas as pd
    #from mpmath import gamma
    #use step as seed here
    np.random.seed(seed)

    #############################################
    # NPS score prediction:
    
    # get features
        
    # convert from days to minutes
    y = y*(24*60) 
    
    # log transform with offset of 1
    log_throughputtime = np.log(1+y)
    
    #get case topic index
    #case_topicidx = c_topic #sigma["c_topic"]
    
    def MatchCategorical(index, levels= ["j_1",
                                           "z_3",
                                           "q_3",
                                           "z_2",
                                           "r_2",
                                           "z_4",
                                           "d_2",
                                           "w_2",
                                           "g_1",
                                           "w_1"]):
                        
            #get value as string
            value = levels[index]
            return value       
    
    #define list of case topics in indexed order
    case_topics = ["j_1",
                   "z_3",
                   "q_3",
                   "z_2",
                   "r_2",
                   "z_4",
                   "d_2",
                   "w_2",
                   "g_1",
                   "w_1"]
    
    #get casetopic as string
    casetopic = case_topics[case_topicidx]
    
    #case topic conditional coefficients from the model
    
    
    if casetopic == "d_2":
        d_2 = 0
        g_1 = 0
        j_1 = 1
        q_3 = 0
        r_2 = 0
        w_1 = 0
        w_2 = 0
        z_2 = 0
        z_3 = 0
        z_4 = 0
     
    if casetopic == "g_1":
        d_2 = 0
        g_1 = 1
        j_1 = 0
        q_3 = 0
        r_2 = 0
        w_1 = 0
        w_2 = 0
        z_2 = 0
        z_3 = 0
        z_4 = 0
        
    if casetopic == "j_1":
        d_2 = 0
        g_1 = 0
        j_1 = 1
        q_3 = 0
        r_2 = 0
        w_1 = 0
        w_2 = 0
        z_2 = 0
        z_3 = 0
        z_4 = 0
        
    if casetopic == "q_3":
        d_2 = 0
        g_1 = 0
        j_1 = 0
        q_3 = 1
        r_2 = 0
        w_1 = 0
        w_2 = 0
        z_2 = 0
        z_3 = 0
        z_4 = 0
        
    if casetopic == "r_2":
        d_2 = 0
        g_1 = 0
        j_1 = 0
        q_3 = 0
        r_2 =1
        w_1 = 0
        w_2 = 0
        z_2 = 0
        z_3 = 0
        z_4 = 0
        
    if casetopic == "w_1":
        d_2 = 0
        g_1 = 0
        j_1 = 0
        q_3 = 0
        r_2 = 0
        w_1 = 1
        w_2 = 0
        z_2 = 0
        z_3 = 0
        z_4 = 0
        
    if casetopic == "w_2":
        d_2 = 0
        g_1 = 0
        j_1 = 0
        q_3 = 0
        r_2 = 0
        w_1 = 0
        w_2 = 1
        z_2 = 0
        z_3 = 0
        z_4 = 0
        
    if casetopic == "z_2":
        d_2 = 0
        g_1 = 0
        j_1 = 0
        q_3 = 0
        r_2 = 0
        w_1 = 0
        w_2 = 0
        z_2 = 1
        z_3 = 0
        z_4 = 0
        
    if casetopic == "z_3":
        d_2 = 0
        g_1 = 0
        j_1 = 0
        q_3 = 0
        r_2 = 0
        w_1 = 0
        w_2 = 0
        z_2 = 0
        z_3 = 1
        z_4 = 0
        
    if casetopic == "z_4":
        d_2 = 0
        g_1 = 0
        j_1 = 0
        q_3 = 0
        r_2 = 0
        w_1 = 0
        w_2 = 0
        z_2 = 0
        z_3 = 0
        z_4 = 1
        
        
    # model params
    intercept = 2.300587
    gammascale = 1.300057
    
    if bias !=0:
        intercept = intercept + bias
    
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
    
    #residuals (exponential regression)
    #residual = -np.log(1-uniform_values[step])
        
    #prediction equation
    NPS = np.exp(intercept + np.dot(X,betas))/gammascale #*float(gamma(1+gammascale))
        
    #add random component
    NPS = np.random.gamma(shape=NPS, scale=gammascale, size=1)[0]-1 #minus the offset added to target
    
    # update weighted NPS_priority:
    NPS_priority = np.abs(NPS-7.5)
    
    """
    Implement winsorizing
    
    1) round up the number to 0 decimals
    2) if rounded number < 0, set it to 0
    3) if < 10, set it to 10
    
    """
    
    NPS = int(np.round(NPS,decimals=0))
    if NPS < 0:
        NPS = 0
    if NPS > 10:
        NPS = 10
    
    
    return NPS, NPS_priority