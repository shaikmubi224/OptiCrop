import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Style
plt.style.use("fivethirtyeight")

# Load dataset
data = pd.read_csv("dataset/crop_recommendation.csv")

# Create output folder
os.makedirs("analysis/graphs/bivariate", exist_ok=True)

# Features to compare with crop label
features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

# Create boxplots
for feature in features:
    plt.figure(figsize=(14, 6))

    sns.boxplot(
        x="label",
        y=feature,
        data=data
    )

    plt.xticks(rotation=90)
    plt.title(f"{feature} vs Crop")

    plt.tight_layout()

    plt.savefig(f"analysis/graphs/bivariate/{feature}_vs_crop.png")

    plt.close()

print("✅ Bivariate Analysis Completed Successfully!")
print("Graphs saved in analysis/graphs/bivariate/")