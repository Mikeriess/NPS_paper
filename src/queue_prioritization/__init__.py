"""
Queue Prioritization Simulation package.

This package provides tools for simulating and analyzing different queue prioritization 
strategies in customer service scenarios.
"""

import os

# Create experiments directory if it doesn't exist
experiments_dir = os.path.join(os.getcwd(), "experiments")
if not os.path.exists(experiments_dir):
    os.makedirs(experiments_dir)

__version__ = "0.1.0" 