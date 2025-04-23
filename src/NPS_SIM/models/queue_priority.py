import numpy as np
np.random.seed(seed)
import pandas as pd
import random


""" Only cases that arrived up until time z + 15 is visible """


def FCFS(Case_DB):
    """
    First-come first served queue management
    """

    """ Sort cases by their arrival time q"""
    #Theta_ordered = sorted(Theta.copy(), key=lambda d: d['q']) 
    
    #sort case db by arrival time
    Case_DB = Case_DB.sort_values("arrival_q",ascending=True)
    
    #update the index
    Case_DB.index = list(range(0,len(Case_DB)))
    
    #set the queue order (including cases that are not in the queue)
    Case_DB["queue_order"] = list(range(0,len(Case_DB)))

    return Case_DB

def SIRO(Case_DB):
    """
    Service in random order 
    """

    #Shuffle the dataframe
    Case_DB = Case_DB.sample(frac = 1)
    
    #Update the indexes
    Case_DB.index = list(range(0,len(Case_DB)))
    Case_DB["queue_order"] = list(range(0,len(Case_DB)))

    return Case_DB

def SRTF(Case_DB):
    """
    Shortest remaining time first queue management
    """

    #sort case db by the predicted throughput time
    Case_DB = Case_DB.sort_values("est_throughputtime", ascending=True)
    
    #change the index
    Case_DB.index = list(range(0,len(Case_DB)))

    #set the queue order (including cases that are not in the queue)
    Case_DB["queue_order"] = list(range(0,len(Case_DB)))

    return Case_DB


def LRTF(Case_DB):
    """
    Longest remaining time first queue management
    """

    #sort case db by the predicted throughput time
    Case_DB = Case_DB.sort_values("est_throughputtime", ascending=False)

    #change the index
    Case_DB.index = list(range(0,len(Case_DB)))

    #set the queue order (including cases that are not in the queue)
    Case_DB["queue_order"] = list(range(0,len(Case_DB)))

    return Case_DB  

def NPS_priority(Case_DB):
    """
    NPS-based priority queue management

    If there is a tie, the one that arrived first will get first served
    """

    #sort case db by arrival time
    Case_DB = Case_DB.sort_values("arrival_q",ascending=True)
    Case_DB.index = list(range(0,len(Case_DB)))
    
    #sort case db by NPS_priority
    Case_DB = Case_DB.sort_values("est_NPS_priority",ascending=True)
    Case_DB.index = list(range(0,len(Case_DB)))
    
    #set the queue order (including cases that are not in the queue)
    Case_DB["queue_order"] = list(range(0,len(Case_DB)))

    return Case_DB
