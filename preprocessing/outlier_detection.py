import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# FiveThirtyEight style
plt.style.use("fivethirtyeight")

# Load dataset
data = pd.read_csv("dataset/crop_recommendation.csv")

# Create folder
os.makedirs("preprocessing/graphs", exist_ok=True)

# Numerical columns
columns = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

print("=" * 60)
print("OUTLIER DETECTION USING IQR")
print("=" * 60)

for column in columns:

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower) |
        (data[column] > upper)
    ]

    print(f"{column}")
    print(f"Lower Bound : {lower:.2f}")
    print(f"Upper Bound : {upper:.2f}")
    print(f"Outliers    : {len(outliers)}")
    print("-" * 50)

    plt.figure(figsize=(8,4))

    sns.boxplot(x=data[column])

    plt.title(f"Boxplot of {column}")

    plt.tight_layout()

    plt.savefig(f"preprocessing/graphs/{column}_boxplot.png")

    plt.close()

print("\n✅ Outlier Detection Completed Successfully!")