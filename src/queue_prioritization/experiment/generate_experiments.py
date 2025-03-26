from helper_functions import *
import numpy as np
import pandas as pd

# Updated imports
from queue_prioritization.experiment.DoE import build_full_fact, fix_label_values


#Simple setup for testing purposes:
Settings = {'F_priority_scheme':["FCFS","NPS"],#,"NPS"],     #
            "days":[365],#,365],
            'F_number_of_agents':[#1,
                                  #3,
                                  3, 
                                  5, 
                                  10, 
                                  15], 
                                  #25, 
                                  #30, 
                                  #35], 
                                  #40],   #
            'Repetition':list(range(0,10))}                #

# Generate a full factorial:
df=build_full_fact(Settings)  


# Constants affecting the experiments:
df["Done"] = 0 #Flag if the final model has been trained
#df["Simulation_duration_sec"] = 0.0


df["RUN"] = df.index + 1
#df["Done"] = 0

print(df)

# Save the settings to a file
np.save('../results/Experiment_Settings.npy', Settings) 

#store the new experimental design
df.to_csv("../results/Experiments.csv", index=False)