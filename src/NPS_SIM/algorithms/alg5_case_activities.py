# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 14:05:10 2022

@author: Mike
"""

import pandas as pd
import numpy as np
#np.random.seed(1337)



"""
DEBUG INITIALIZE WITH CONTROLLED VALUES


Theta = [{'i': 19, 'q': 10.50, 'c': ['a', 'b', 'c'], 'r':[1], 'j':[],'a':[],'t':[]},
         {'i': 20, 'q': 10.99, 'c': ['a', 'b', 'c'], 'r':[2], 'j':[],'a':[],'t':[]}]

Psi = [{'id': 0, 'i': [19], 'q': 0.0, 'status': 'assigned'},
       {'id': 1, 'i': [20], 'q': 0.1, 'status': 'assigned'}]

L = []

z = 10

z_window_start = 10
window_end = 11
"""

def CaseActivities(d, Case_DB, Theta, Psi, L, p_vectors, seed, start_delays=True, end_delays=True, verbose=False, counter=0):
    import datetime
    
    # Set a consistent random seed at the beginning of the function
    # This ensures reproducibility while keeping all random decisions deterministic
    np.random.seed(seed + counter)
    
    """
    ###########################################################################
    """
    
   
        
    """ Define functions """
    
    def activity_dist(k, a_k0, counter, p_vectors):
        """
        This should contain both the probability distribution of the unconditional
        inputs, as well as the duration model equation
        """
                
        """ implement markov chain here """
        
        if verbose == True:
            print(k,a_k0)
                
        
        """
        p0 = [0.0, 0.92, 0.08, 0.0]
        
        P_a1 = [0.08, 0.00, 0.67, 0.25]
        P_a2 = [0.00, 0.02, 0.96, 0.02]
        P_a3 = [0.00, 0.02, 0.45, 0.53]
        P_a4 = [0.00, 0.00, 0.00, 1.00]
        """
        
        #define the states
        states = ["Task-Reminder", "Interaction", "Email", "END"]
        
        if a_k0 == "start":
            #a = np.random.choice(states, size=1, replace=False, p=p0)[0]
            
            
            a = p_vectors["p0"][counter]
            
        else:
            if a_k0 == states[0]:
                #a = np.random.choice(states, size=1, replace=False, p=P_a1)[0]
                a = p_vectors["p1"][counter]
            if a_k0 == states[1]:
                #a = np.random.choice(states, size=1, replace=False, p=P_a2)[0]
                a = p_vectors["p2"][counter]
            if a_k0 == states[2]:
                #a = np.random.choice(states, size=1, replace=False, p=P_a3)[0]
                a = p_vectors["p3"][counter]
            if a_k0 == states[3]:
                #a = np.random.choice(states, size=1, replace=False, p=P_a4)[0]
                a = p_vectors["p4"][counter]
                
        ########################
        
        next_activity = a
        
        next_timestep = k +1
        
        counter = counter +1
        
        return next_activity, next_timestep, counter


    def duration_dist(y, y_dt, k, a, psi_personality, case_topic, counter):
        """
        This should contain both the probability distribution of the unconditional
        inputs, as well as the duration model equation
        """
        import numpy as np
        #from mpmath import gamma
        import datetime
        
        # Use a deterministic seed based on the main seed, counter, and step k
        # This ensures reproducibility for each duration calculation
        local_seed = seed + counter + k
        np.random.seed(local_seed)
        
        ############
        # general params
        intercept = 1.6645
        scale_theta = 0.3908
        
        ############
        # features:
        task_number = k
        task_number_effect = 0.0420
        
        personality_effect = psi_personality #sampled from distribution X
        
        ####### Case topic
        """
        ordered_case_topic_betas = [
                      -0.073698888,  #0: j_1 
                       0.1672023625, #1: z_3
                       0.133908834,  #2: q_3
                      -0.025270165,  #3: z_2
                       0.0512409836, #4: r_2
                       0.0000000000, #5. z_4 REFERENCE
                       0.0184041223, #6: d_2
                       0.000000000,  #7: w_2
                      -0.015895652,  #8: g_1
                      -0.041859451]  #9: w_1
        """
        ordered_case_topic_betas = [
                      -0.0557,  #0: j_1  #
                       0.1637, #1: z_3
                       0.1712,  #2: q_3
                      -0.0420,  #3: z_2
                       0.0836, #4: r_2
                       0.0000, #5. z_4 REFERENCE
                       0.0200, #6: d_2
                       0.0119,  #7: w_2
                      -0.0538,  #8: g_1
                      -0.0609]  #9: w_1
                         
        case_topic_effect = ordered_case_topic_betas[case_topic]
        
        ####### Activity
        
        if a == "Interaction":
            activity_effect =  0.1057 
        if a == "Email":
            activity_effect =  0.0180
        if a == "Task-Reminder":
            activity_effect = 0.0
        if a == "END":
            activity_effect = 0.0
        
        
        Beta = [case_topic_effect,
                personality_effect,
                activity_effect,
                task_number_effect]
        
        X = [1,
             1, 
             1,
             task_number]
        
        # prediction model
        #duration = np.exp(intercept + np.dot(X,Beta))*float(gamma(1+theta))
        
        linear_comb = np.exp(intercept + np.dot(X,Beta))
        
        duration = np.random.weibull(linear_comb)*scale_theta
        
        """#####################  MANIPULATED VALUES ####################### """
        
        t = duration#0.1
        
        t_start = y
        t_end = y + t
        
        t_start_dt = y_dt
        t_end_dt = y_dt + datetime.timedelta(days=t)
        
        return t_start, t_end, t, t_start_dt, t_end_dt
    
    def GenerateStartDelay(time):
        # Reset random seed for this function to ensure reproducibility in delay generation
        # Using a different seed modification to avoid correlations
        np.random.seed(seed + int(time*100))
                    
        """
        ###########################################################
        """
        
        # Get day of week [1:7]
        weekday_number = int(np.mod(time,7) + 1) #time_dt.weekday()+1
        
        if verbose == True:
            print("weekday:",weekday_number,"                   (time+1 to get [1:7] range)")
            print("planned start:",time)
        
        # Get decimal value only, to see time of day
        time_of_day = np.mod(time,1) 
        
        if verbose == True:
            print("time of day:",time_of_day)
            
        """
        ###########################################################
        """
        
        # Initialize the delay
        t_start_delay = 0
        
        
        # first type of delay: weekends are closed for business
        if weekday_number >= 6:
            time_to_monday = 8 - weekday_number #move time to monday morning
            t_start_delay = time_to_monday + 0.33334 #move time to monday at 8
            if verbose == True:
                print("start-delay added:",t_start_delay,"(",t_start_delay+time,")","type:","weekday","original:",time)
            
        # second type of delay: outside opening hours
        else:
        
            if time_of_day >= 0.75: #if at or after 18:00
                time_to_midnight = 1 - time_of_day #get time to midnight
                time_to_open = time_to_midnight + 0.33334 #get time from midnight to open at 8:00
                t_start_delay = t_start_delay + time_to_open #add the delay
                if verbose == True:
                    print("start-delay added:",t_start_delay,"(",t_start_delay+time,")","type:","over 0.75","original:",time)
                
            
            if time_of_day < 0.33333: #if it is before 8:00 in the morning at a business day
                time_to_open = 0.33334 - time_of_day #get the time until 8:00
                t_start_delay = t_start_delay + time_to_open #add the delay
                if verbose == True:
                    print("start-delay added:",t_start_delay,"(",t_start_delay+time,")","type:","under 0.3333","original:",time)
        
        # final output
        time_delay = t_start_delay
        
        ######################################################
        #second check: is the delayed time in the weekend?
        new_time = time + time_delay
        
        # Get day of week [1:7]
        new_weekday_number = int(np.mod(new_time,7) + 1) #time_dt.weekday()+1
        
        
        # first type of delay: weekends are closed for business
        if new_weekday_number >= 6:
            
            if verbose == True:
                print("new weekday:",new_weekday_number,"                   (time+1 to get [1:7] range)")
                print("new planned start:",new_time)
            
            time_to_monday = 8 - new_weekday_number #move time to monday morning
            new_t_start_delay = time_to_monday + 0.33334 #move time to monday at 8
            
            if verbose == True:
                print("start-delay added:",new_t_start_delay,"(",new_t_start_delay+time,")","type:","weekday","original:",new_time)
            
            #override the existing delay with further delay
            time_delay = new_t_start_delay

        return time_delay
    
    
    
    """
    ###########################################################################
    """
    
    """ Initialize variables """
    window_end = d + 1
    
    
    """ for each agent"""
    for agentidx in range(0,len(Psi)):
        
        """ get current agent """
        psi = Psi[agentidx].copy()
        
        """ if assigned to a case"""
        if len(psi["i"]) > 0:
            
            if verbose == True:
                print("agent:",psi["id"],"is assigned to case",psi["i"])
            
            
            """ get the case id"""
            caseid = psi["i"][0]
            
            """ get the active case"""
            sigma = Theta[caseid].copy()
            
            """ get the time the agent is available"""
            q = psi["q"]
            q_assigned = psi["q_assigned"]
            #print("Agent q-value:",q)
            
            """ get last activity, its finish time, and next timestep k"""
            
            
            if len(sigma["j"]) == 0:
                first_event = True
                a_k0 = "start"
                k = 0
                
                """
                
                # q is initialized by 0, so cases cannot start with 0
                and must then start from the assignment time q_assigned
                
                # if q_assgined is less than the current availability,
                activity should start with q instead
                
                """
                
                # new approach: When the agent became available
                agent_availability = np.max([q, q_assigned])
                
                t_start = agent_availability
                t_start_dt = datetime.datetime(2018, 1, 1, 0, 0, 0) + datetime.timedelta(days=t_start)
                
                """ Case is now given a topic when agent reads the email/text"""
                                   
                Theta[caseid]["c_topic"] = p_vectors["c_topic"][counter]
                
            else:
                first_event = False
                k = len(sigma["j"]) #+ 1
                a_k0 = sigma["a"][-1]
                
                t_start = sigma["t_end"][-1]
                t_start_dt = sigma["t_end_dt"][-1]
            
            
            
            """ while y is still within the current 15-minute window """
            while t_start < window_end:
                
                """
                ############################################################
                OFFICE HOURS: PART 1 - Starting the work inside office hours
                ############################################################
                """
                
                if verbose == True:
                    print("="*50)
                    print("case:",caseid)
                
                
               
                # testing the function
                t_start_delay = GenerateStartDelay(time = t_start)
                #t_start_delay = 0 ########## Temporary bypass
                
                
                """ Avoid continuing if last event has happened """
                               
                if a_k0 == "END":
                    break
                
                    if verbose == True:
                        print("breaking since, a_k0 is END")
                    
                """
                ############################################################
                Generating the activities and durations
                ############################################################
                """
                    
                # Get the next activity
                a_k, k, counter = activity_dist(k, a_k0, counter=counter, p_vectors=p_vectors) #previous A_k
                
                # update t_start with start delay if any
                t_start = t_start + t_start_delay
                t_start_dt = t_start_dt + datetime.timedelta(days=t_start_delay)
                
                # second check of the delay: did the rescheduled date also conform to the rules?
                
                if verbose == True:
                    print("t_start:",t_start," - ", t_start_dt.strftime('%m/%d/%Y %H:%M:%S'))
                # Get the duration of next activity
                t_start, t_end, t, t_start_dt, t_end_dt = duration_dist(t_start, 
                                                                        t_start_dt,
                                                                        k=k, 
                                                                        a=a_k,
                                                                        psi_personality=psi["personality"], 
                                                                        case_topic=Theta[caseid]["c_topic"],
                                                                        counter=counter) 
                
                if verbose == True:
                    print("duration:",t)
                    print("t_end:",t_end)                   
                    print("simulated values:",k, a_k, t_start, t_end,"duration:",t)


                """
                ############################################################
                OFFICE HOURS: PART 2 - Ending the work inside office hours
                ############################################################
                """
                
                t_end_delay = 0 ########## Not included in this version

                                                    
                """ append the updated trace back into Theta """
            
                
                # append delays
                Theta[caseid]["t_start_delay"].append(t_start_delay)
                Theta[caseid]["t_end_delay"].append(t_end_delay)
                
                
                # update activity
                Theta[caseid]["a"].append(a_k)
                
                # update timestep
                Theta[caseid]["j"].append(k)   
                
                """# update start time: First event time values done in case assignment
                #if first_event == False:"""
                Theta[caseid]["t_start"].append(t_start)
                Theta[caseid]["t_start_dt"].append(t_start_dt)
            
               
                # update end time: same for all timesteps
                Theta[caseid]["t_end"].append(t_end)
                Theta[caseid]["t_end_dt"].append(t_end_dt)
                
                # update current event duration
                Theta[caseid]["t"].append(t)
                
                # update current event delays
                Theta[caseid]["t_delayed"].append(t_start_delay + t_end_delay)
                
                
                # Update the first event flag
                first_event = False
                
               
                
                
                """ update timer y """
                #y = y + t
                t_start = t_end
                t_start_dt = t_end_dt #t_start_dt + datetime.timedelta(days=t_start_delay)
                
                if verbose == True:
                    print("updated timer, t_start is now:",t_start)
                
                """ update the agent pool """
                if a_k == "END":
                    
                    """ if this is the last activity, unassign agent, update last 
                    activity time and remove case from the case pool theta """
                    
                    if verbose == True:
                        print("end of case",caseid)
                    
                    # remove the case id from agent object, if case is done
                    Psi[agentidx]["i"] = []
                    
                    # update agent status
                    Psi[agentidx]["status"] = "idle"
                    
                    # update agent idle time
                    Psi[agentidx]["q"] = t_end
                    
                    if verbose == True:
                        print("agent ready for next case at time:",t_end)
                    
                    # remove the case id from the case pool theta
                    Theta[caseid]["status"] = "closed"
                    #Theta[:] = [j for j in Theta if j.get('i') != caseid]
                    
                    # calculate the initial delay: The time between the arrival and the work has started
                    Theta[caseid]["initial_delay"] = Theta[caseid]["t_start"][0] - Theta[caseid]["q"]
                    
                    """ append the finalized trace to the event-log L """
                    L.append(Theta[caseid].copy())
                                                                              
                    
                    """ UPDATE THE CASE DB """
                    #get the row index in case database for the current case in theta
                    casedb_idx = Case_DB.index[Case_DB['theta_idx'] == caseid].tolist()[0]
                    
                    #update that row of casedb such that no further work is done on this case
                    Case_DB.at[casedb_idx, 'case_queued'] = False
                    Case_DB.at[casedb_idx, 'status'] = "closed"
                    
                    """ break out of the while loop"""
                    break
                
                # update last activity end time for the agent
                Psi[agentidx]["q"] = t_end
                
                
    
    return L, Case_DB, Theta, Psi, counter
