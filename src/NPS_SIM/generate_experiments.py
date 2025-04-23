# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 09:53:21 2022

@author: Mike
"""


"""
Experiment settings
"""

from experiment.DoE import build_full_fact, fix_label_values
import os

             
run_settings = {'F_priority_scheme':["NPS","SRTF","LRTF","FCFS"], 
                
                # number of queues
                'F_number_of_agents':[3], #, 4, 5, 6, 7, 8, 9
                
                # hard ceiling
                'F_hard_ceiling':["NONE"],#["NONE","SLA"],
                
                # waiting time until ceiling takes effect
                "F_ceiling_value":[2.5],      
                                                
                # how many days are used as burn-in (can be zero)
                "F_burn_in":[0],#,365],
                
                # how many days to simulate
                "F_days":[50],#,365],
                
                # bias to the prior NPS-distribution 
                "F_NPS_dist_bias":[0], # bias to the prior NPS-distribution

                # multiplier (!) for the effect of waiting time on NPS
                "F_tNPS_wtime_effect_bias":[1.0], # 1 = no change, >1 = increase, <1 = decrease
                
                # date from which to start simulation: YYYY-MM-DD 
                "startdate":["2018-07-01"],
                                          
                # Repeated runs
                "repetition":list(range(0, 1))} #100


# Generate a full factorial:
df = build_full_fact(run_settings)

# Get string values back
df = fix_label_values(df, run_settings, variables = ["F_priority_scheme","F_hard_ceiling","startdate"])

# if burn in period is incorrectly specified, increase days by burn-in period
for day in df.index:
    if df.loc[day, "F_burn_in"]  <= df.loc[day, "F_days"]:
        df.loc[day, "F_days"] = df.loc[day, "F_days"] + df.loc[day, "F_burn_in"]



"""
Ensure dtypes are correct
"""

#change dtypes
df.F_number_of_agents = df.F_number_of_agents.astype(int)
df.repetition = df.repetition.astype(int)
df.F_days = df.F_days.astype(int)
df.F_burn_in = df.F_burn_in.astype(int)

"""
Placeholder variables: -1 for unobserved
"""
df['closed_avg_simulated_NPS'] = -1
df['closed_avg_simulated_throughput_time'] = -1
df['closed_avg_predicted_NPS'] = -1
df['closed_avg_predicted_throughput_time'] = -1
df['closed_avg_predicted_NPS_priority'] = -1    
df['closed_avg_initial_delay'] = -1
df['closed_avg_activity_start_delay'] = -1
df['closed_avg_duration_delayed'] = -1
df['all_avg_simulated_NPS'] = -1
df['all_avg_simulated_throughput_time'] = -1
df['all_avg_predicted_NPS'] = -1
df['all_avg_predicted_throughput_time'] = -1
df['all_avg_predicted_NPS_priority'] = -1    
df['all_avg_initial_delay'] = -1
df['all_avg_activity_start_delay'] = -1
df['all_avg_duration_delayed'] = -1
df['cases_arrived'] = -1
df['cases_closed'] = -1
df['case_queued'] = -1
df['cases_assigned_at_end'] = -1
df['min_tracelen'] = -1
df['max_tracelen'] = -1
df['Simulation_duration_min'] = -1

"""
sort experiments by dataset
"""

df["RUN"] = list(range(0,len(df)))
df["Done"] = 0

print(df)

"""
Save experiments table
"""

path = "results/"
if os.path.isdir(path) == False:
    #if it does not exist: create folder for the results
    os.mkdir(path)

df.to_csv("results/design_table.csv",index=False)

