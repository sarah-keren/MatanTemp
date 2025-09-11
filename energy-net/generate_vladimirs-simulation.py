import pandas as pd
import numpy as np

# 1. Load the table (HOUR, SMP, B1..B6).  SMP will be ignored.
df = pd.read_csv("PCS-Units.csv")    # columns: HOUR,SMP,B1,B2,B3,B4,B5,B6

# 2. For each battery column, create a (T,1) array and save to an .npy file.
for col in ["B1", "B2", "B3", "B4", "B5", "B6"]:
    actions = df[col].fillna(0).to_numpy().reshape(-1, 1)  # + => charge, – => discharge
    np.save(f"{col.lower()}_actions.npy", actions)
