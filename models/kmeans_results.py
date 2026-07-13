import pandas as pd
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("dataset/crop_recommendation.csv")

# Features
X = data.drop("label", axis=1)

# Train K-Means
kmeans = KMeans(
    n_clusters=4,
    init="k-means++",
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X)

# Add cluster column
result = data.copy()
result["Cluster"] = clusters

print("=" * 60)
print("K-MEANS CLUSTER RESULTS")
print("=" * 60)

for i in range(4):
    print(f"\nCluster {i}")
    print(result[result["Cluster"] == i]["label"].unique())