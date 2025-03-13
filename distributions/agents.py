def generate_agents(agents, seed):
    import numpy as np

    np.random.seed(seed)
    
    """
    Generate the agents
    """
            
    Psi = []
    
    # Normal distribution
    agent_personalities = np.random.normal(loc=0.2171,
                         scale=0.5196,
                         size=agents)
    
    for a in range(0,agents):
        psi = {"id":a,
                 "i":[],
                 "q":np.round(a*0.1, decimals=1), # debugging only
                 
                 "personality":agent_personalities[a],
                 
                 "status":None}
        
        Psi.append(psi)
        
    return Psi