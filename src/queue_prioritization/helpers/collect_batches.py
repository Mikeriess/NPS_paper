import os

# Define experiments directory
EXPERIMENTS_DIR = os.path.join(os.getcwd(), "experiments")

# Then update all file paths
# For example:
# Original: exp = pd.read_csv("results/design_table.csv")
# Updated: exp = pd.read_csv(os.path.join(EXPERIMENTS_DIR, "design_table.csv"))

# Original: evlog = pd.read_csv("results/"+str(RUN)+"/"+str(RUN)+"_log.csv")
# Updated: evlog = pd.read_csv(os.path.join(EXPERIMENTS_DIR, str(RUN), f"{RUN}_log.csv")) 