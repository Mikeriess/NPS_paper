
def sim_generalized_lognormal(n=1):
    import numpy as np
    
    #simulate from normal
    x = np.random.normal(loc=0.2835587,#-3.228792,
                         scale=0.7866424,#2.0041058,
                         size=n)[0]
    
    #generalized lognormal
    shape = 0.0302774
    
    if shape > 0:
        #transform into generalized lognormal
        x = (np.power(x,shape)-1)/shape
    else:
        #alternative transformation
        x = np.log(x)
    
    return x