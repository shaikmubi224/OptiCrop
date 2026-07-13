import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("dataset/crop_recommendation.csv")

# Features
X = data.drop("label", axis=1)

# Create output folder
os.makedirs("models/graphs", exist_ok=True)

# Elbow Method
wcss = []

for i in range(1, 11):
    model = KMeans(
        n_clusters=i,
        init="k-means++",
        max_iter=300,
        n_init=10,
        random_state=42
    )

    model.fit(X)
    wcss.append(model.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8,5))

plt.plot(range(1,11), wcss, marker="o")

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.tight_layout()

plt.savefig("models/graphs/elbow_method.png")

plt.close()

print("✅ Elbow Method Completed.")