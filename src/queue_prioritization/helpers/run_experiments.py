import os

# Define experiments directory
EXPERIMENTS_DIR = os.path.join(os.getcwd(), "experiments")

# Then update all file paths
# For example:
# Original: path = "results/"+str(RUN)
# Updated: path = os.path.join(EXPERIMENTS_DIR, str(RUN)) 