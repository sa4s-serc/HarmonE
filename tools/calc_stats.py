import pandas as pd
df = pd.read_csv("output/pred_switching_retrain.csv")  # Replace with your actual file path
print(df.head)
stats = df.groupby("model_used")["energy"].describe()
print(stats)
