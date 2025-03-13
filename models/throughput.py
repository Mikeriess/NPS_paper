def predict_TT_Notopic(sigma):
    """
    Throughput time prediction without case topic info
    """
     
    import numpy as np
    #import pandas as pd
    #from mpmath import gamma
    
    """
    ################################################
    Remaining time prediction 
    ################################################
    """
    
    # get features 
    dt = sigma["q_dt"]
    
    # #np.mod(z,7)
    day = dt.day
    
    year = dt.year
    month = dt.month
    weekday = dt.weekday()
    hour = dt.hour
    
    # predict throughput time from scoring equation
    
    betas = [-0.02950, #year
             -0.04149, #month
             -0.00034, #weekday
              0.06511] #hour
    
    X = [year,
         month,
         weekday,
         hour]
    
    c = 66.80872
    
    
    #prediction expression (-1 since original target is +1)
    y = np.exp(c + np.dot(X,betas))-1
    
    #simulation expression
    #scale = 2.2944
    # y = np.random.normal(loc=linear_comb,
    #                      scale=scale,
    #                      size=1)[0]
    
    #convert from minutes to days
    y = y/60/24 
    
    # update case with estimated throughput time
    sigma["est_throughputtime"] = y
    return sigma






def predict_TT(sigma):
    """
    Throughput time prediction with case topic info
    """
     
    import numpy as np
    #import pandas as pd
    #from mpmath import gamma
    
    date_and_time = sigma["q_dt"]
    case_topicidx = sigma["c_topic"]
    
     # Inputs for case arrival model
    hour = date_and_time.hour
    weekday = date_and_time.weekday() #np.mod(z,7)
    month = date_and_time.month
    day = date_and_time.day
    year = date_and_time.year
    
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
        
    
    """
    ################################################
    Throughput time prediction 
    ################################################
    """
        
    # model params
    intercept = 139.3817
  
    #coefficients
    betas = [-0.0654, #year
             -0.0495, #month
             0.0077, #day
             0.0197, #weekday
             0.0602, #hour   
             
             -0.3368, #d2
             0.0, #g1
             -1.2246, #j1
             0.2251, #q3
             1.1927, #r2
             -1.1476, #w1
             0.2342, #w2
             -0.1040, #z2
             0.1695, #z3
             0.0 #z4
             ]
    
    #inputs
    X = [year,
         month,
         day,
         weekday,
         hour,
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
    y = np.exp(intercept + np.dot(X,betas)) #*residual
        
    
    #convert from minutes to days
    y = y/60/24 
    
    # update case with estimated throughput time
    sigma["est_throughputtime"] = y
    
    
    return sigma
