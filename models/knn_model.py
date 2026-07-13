import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 60)
print("K-NEAREST NEIGHBORS (KNN)")
print("=" * 60)

# Load Dataset
df = pd.read_csv("dataset/crop_recommendation.csv")

# Features and Target
X = df.drop("label", axis=1)
y = df["label"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Create KNN Model
model = KNeighborsClassifier(n_neighbors=5)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy : {accuracy:.4f}")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

# Save Model
pickle.dump(model, open("saved_models/knn_model.pkl", "wb"))

print("\n✅ KNN Model Saved Successfully!")
