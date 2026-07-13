import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("dataset/crop_recommendation.csv")

# Features
X = data.drop("label", axis=1)

# Target
y = data["label"]

print("=" * 60)
print("DATASET SPLITTING")
print("=" * 60)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data")
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)