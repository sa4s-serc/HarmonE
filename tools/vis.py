import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../knowledge/predictions.csv")
df["cumulative_avg_energy"] = df["energy"].expanding().mean()
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["cumulative_avg_energy"], label="Cumulative Avg Energy", color="blue")
plt.axhline(y=40000, color="red", linestyle="dotted", label="Threshold (40,000)")

plt.xlabel("Entry Index")
plt.ylabel("Cumulative Average Energy")
plt.title("Cumulative Average Energy Usage Over Time")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.savefig("hello.png")
