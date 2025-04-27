# Context Queue-prioritization

The study aim to compare different queue prioritization methods in a agent-based simulation model which is calibrated from a real process in a telecommunications operator. The study compares FCFS, LRTF, SRTF and a suggested new method called NPS, which prioritizes people in the queue which have previously responded with a passive NPS score. By prioritizing these customers, the algorithm aims to move them from the passives segment into the promoters, based on the assumption that there is a relationship between the throughput time of their issue and their NPS response afterwards. The paper is based on three studies: 1) which calibrates a simulation model, 2) which investigates the effect in the empirical setting and 3) which performs counterfactual sensitivity analysis by modifying the relationship between the NPS and throughput time in the empirical setting.

# Research plan

Plan is to lay the groundwork for Study 3, which will be a scenario analysis of the effect of changing the parameters in the empirical setting: By changing the effect of waiting time on NPS score, and the distribution characteristics of the NPS.

<u>Notes from research meeting:</u>

- The simulated NPS stems from a gamma regression, which defines the conditional distribution of the NPS based on the case throughput time, as well as the case topics.

- We have two variables that we can manipulate in the NPS simulation equation:
  
  - **F_NPS_dist_bias** which is additive, i.e. it is a value added to the intercept (alpha)
  
  - **F_tNPS_wtime_effect_bias** which is multiplicative, i.e. it scales the size of the case throughput time, which is also log transformed (beta)
  
  - After the gamma regression, the simulated NPS response is winsorized, such that it falls within expected range of a NPS response

- What we want to knows is:
  
  - What happens to the distribution when we change the values of the two variables above? 
    
    - Does it increase the variance? Does the responses just move towards a uniform distribution? Will we have to change the denominator in the gamma distribution?
  
  - On a higher level: 
    
    - When does the findings change, when we change the effect of throughput time and NPS score? 
    
    - How much do we need to change it to see a change in the results?



# Workflow Notes

- Cursor rules + using the reasoning models seem to improve the usefullness

- Idea: 
  
  - Have experiments be defined from a json file, such that the codebase does not need to be touched
  
  - Have the LLM create a markdown document before running an experiment - the markdown should outline what parameters to set, which quyestions we have that should be revealed by the experiment, and what hypotheses we have prior to running the experiment. In this way, the procedure will be very well documented, easier to write up and likely more interesting for the reader.


