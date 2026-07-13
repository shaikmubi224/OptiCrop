import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style
plt.style.use("fivethirtyeight")

# Load dataset
data = pd.read_csv("dataset/crop_recommendation.csv")

# Create folder
os.makedirs("analysis/graphs/multivariate", exist_ok=True)

# Correlation matrix
plt.figure(figsize=(10, 8))

correlation = data.drop("label", axis=1).corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="YlGnBu",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("analysis/graphs/multivariate/correlation_heatmap.png")

plt.close()

print("✅ Multivariate Analysis Completed Successfully!")
print("Graph saved in analysis/graphs/multivariate/")