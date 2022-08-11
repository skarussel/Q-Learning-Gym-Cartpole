from email import header
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv", delimiter="\t")
print(df)
df.plot.line("i_episode", "Reward")
plt.show()
