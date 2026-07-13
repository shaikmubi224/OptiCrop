import pandas as pd

# Load dataset
data = pd.read_csv("dataset/crop_recommendation.csv")

print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(data.head())

print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(data.shape)

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
data.info()

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(data.isnull().sum())