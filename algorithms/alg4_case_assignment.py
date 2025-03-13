# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 14:57:40 2022

@author: Mike
"""

import numpy as np
#np.random.seed(1337)
import pandas as pd


#Theta_ordered = [{'i': 20, 'q': 10.758166076206798, 'c': ['a', 'b', 'c']},
#         {'i': 19, 'q': 10.3790830381034, 'c': ['a', 'b', 'c']}]

"""
Generate the agents


Psi = []

for a in range(0,10):
    
    psi = {"id":a,
             "i":[],
             "q":a*0.1,
             "status":None}
    
    if a == 2:
        
        psi = {"id":a,
                 "i":[998],
             "q":a*0.1,
             "status":"pre-assigned"}
        
    if a == 3:
        
        psi = {"id":a,
                 "i":[999],
             "q":a*0.1,
             "status":"pre-assigned"}
    
    
    Psi.append(psi)

# Copy the list 
#Psi_copy = Psi.copy()

"""



def CaseAssignment(Case_DB, Theta, Psi, verbose, time_interval, seed=1337):
    import datetime
    np.random.seed(seed)
    
    """ Create a temporary pool for idle agents A"""
    A = []
    
    """ For every agent in the agent pool """
    
    for agentidx in range(0,len(Psi)):
        
        psi = Psi[agentidx].copy()
        if verbose == True:
            print(psi)
        
        """ If agent is not assigned to a case """
        
        if len(psi["i"]) == 0:
            if verbose == True:
                print("moving agent", psi["id"], "to A")
            
            """ Add agent to agent pool A, remove from agent pool"""
            A.append(psi.copy())
            
            """ Sort pool of available agents by their idle time"""
            A = sorted(A, key=lambda d: d['q'], reverse=False) 
            
            #Psi.remove(psi)
        else:
            if verbose == True:
                print("agent occupied...")
    
    # Remove all agents currently in agent pool A from the main pool Psi
    Psi = [x for x in Psi if x not in A]
    
    """
    Sense-making
    """
    
    if verbose == True:
        print("#"*30)
        print("A: available agents =     ",len(A))
        print("Psi: unavailable agents = ",len(A))
        print("#"*30)
    
    
    
    """
    ##########################################################################
    Cases to assign from:
    """
    # Look only in the queue
    Queue = Case_DB.loc[Case_DB.case_queued == True]
    
    # Look only at cases that have arrived
    Queue = Queue.loc[Case_DB.case_currently_arrived == True]
    
    if len(Queue) > 0:
        #print("Queue:",Queue.theta_idx.values)
    
        """
        Assign available agents
        """
        for casedata in Queue.index:
            
            case_i = Queue.loc[casedata]
            
            #get the case id in Theta
            theta_idx = case_i.theta_idx
            
            #get row id in Case_DB
            casedb_idx = case_i.casedb_idx
            
            #subset on a single case
            sigma = Theta[theta_idx]
            
            
            """ if case is unassigned """
            if len(sigma["r"]) == 0:        
                if verbose == True:
                    print("case:",sigma["i"])
                
                agent_assigned = False
                
                for agent in A:
                    
                    if len(agent["i"]) == 0:
                        if verbose == True:
                            print(agent)
                        
                        # Assign agent to case
                        agent["i"] = [sigma["i"]]
                        agent["status"] = "assigned to case"
                        
                        #add time of assignment
                        agent["q_assigned"] = time_interval
                        
                        # Assign case to agent
                        sigma["r"].append(agent["id"])
                        
                        #add time of assignment
                        sigma["q_assigned"] = time_interval
                                            
                        # Update case status
                        sigma["status"] = "assigned"
                        
                        # Flag to stop searching for other agents after first success
                        agent_assigned = True
                                        
                    if agent_assigned == True:
                        """
                        If successfull, update the case DB
                        """
                        
                        Case_DB.at[casedb_idx, 'agent_idx'] = agent["id"]
                                                
                        if verbose == True:
                            print("agent",agent["id"],"assigned..", "time:",time_interval)
                            print(Case_DB.at[casedb_idx, 'agent_idx'])
                            #print("#"*30)
                        """
                        And break out of the loop to stop searching
                        """
                        break
                
    """ Append all agents to agent pool again """
    
    Psi_out = Psi.copy() + A.copy()
    Psi_out = list({v['id']:v for v in Psi_out}.values()) # Remove duplicates, based on id
    
    if verbose == True:
        print("#"*30)
        print("Psi: Outputted agents \n",Psi_out)
        
        print("#"*30)
        print(Case_DB)
    
    return Case_DB, Theta, Psi_out