import pandas as pd

# Load dataset
df = pd.read_csv("data/pems/flow_data_test.csv")
df.columns = df.columns.str.strip()  # Clean column names

# Define sections to delete (start, end)
delete_sections = [(6000, 9000), (8000, 8000), (14501,50000)]

# Remove sections
for start, end in delete_sections:
    df = df.drop(df.index[start:end])

# Reset index after deletion
df = df.reset_index(drop=True)

# Save modified dataset
df.to_csv("data/pems/flow_data_cleaned.csv", index=False)

print("✅ Sections deleted and new file saved as 'flow_data_cleaned.csv'")
