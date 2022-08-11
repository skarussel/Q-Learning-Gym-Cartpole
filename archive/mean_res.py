from email import header
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("mean_results.csv", delimiter=":", header=None, names=["Episode", "Total Mean Reward"])
df.plot.line("Episode", "Total Mean Reward")
plt.show()
