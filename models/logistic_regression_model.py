import os
import pickle
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("dataset/crop_recommendation.csv")

# Features
X = data.drop("label", axis=1)

# Target
y = data["label"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train
model.fit(X_train_scaled, y_train)

# Prediction
y_pred = model.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("="*60)
print("LOGISTIC REGRESSION")
print("="*60)

print(f"\nAccuracy : {accuracy:.4f}")

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, y_pred))

# Save Model
os.makedirs("saved_models", exist_ok=True)

pickle.dump(model,
            open("saved_models/logistic_model.pkl","wb"))

pickle.dump(scaler,
            open("saved_models/scaler.pkl","wb"))

print("\n✅ Logistic Regression Model Saved Successfully!")