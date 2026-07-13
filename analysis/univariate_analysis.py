import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# FiveThirtyEight style (matches your documentation)
plt.style.use("fivethirtyeight")

# Load dataset
data = pd.read_csv("dataset/crop_recommendation.csv")

# Features to analyze
features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

# Create folder to save graphs
os.makedirs("analysis/graphs/univariate", exist_ok=True)

# Plot each feature
for feature in features:
    plt.figure(figsize=(8,5))

    sns.histplot(data[feature], kde=True)

    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(f"analysis/graphs/univariate/{feature}.png")

    plt.close()

print("✅ Univariate Analysis Completed Successfully!")
print("Graphs saved in analysis/graphs/univariate/")