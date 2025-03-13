def generate_p_vectors(seed):
    import numpy as np

    np.random.seed(seed)
    
    """
    Generate simulated activities using same seed
    """
    
    states = ["Task-Reminder", "Interaction", "Email", "END"]
    
    p_vectors = {"p0":np.random.choice(states, size=1000000, replace=True, p=[0.0, 0.92, 0.08, 0.0]),
                 "p1":np.random.choice(states, size=1000000, replace=True, p=[0.08, 0.00, 0.67, 0.25]),
                 "p2":np.random.choice(states, size=1000000, replace=True, p=[0.00, 0.02, 0.96, 0.02]),
                 "p3":np.random.choice(states, size=1000000, replace=True, p=[0.00, 0.02, 0.45, 0.53]),
                 "p4":np.random.choice(states, size=1000000, replace=True, p=[0.00, 0.00, 0.00, 1.00]),
                 "c_topic":np.random.choice(list(range(0,10)), size=1000000, replace=True, p=[0.318670,
                                                                                                0.243781,
                                                                                                0.138256,
                                                                                                0.066510,
                                                                                                0.064415,
                                                                                                0.063891,
                                                                                                0.039277,
                                                                                                0.037968,
                                                                                                0.014925,
                                                                                                0.012307])}
        
    return p_vectors