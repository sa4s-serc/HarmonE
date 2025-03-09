import pandas as pd
import glob

# Path to your CSV files
file_paths = sorted(glob.glob("../data/pems/*.csv"))

# Read and concatenate the 'Flow (Veh/5 Minutes)' column from each file
flow_data = []
for file in file_paths:
    df = pd.read_csv(file, usecols=["Flow (Veh/5 Minutes)"])
    flow_data.append(df)

# Concatenate all data into a single column
merged_df = pd.concat(flow_data, ignore_index=True)

# Save to a single CSV file
merged_df.to_csv("../data/pems/flow_data.csv", index=False, header=False)
