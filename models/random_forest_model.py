import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("RANDOM FOREST")
print("=" * 60)

# Load Dataset
df = pd.read_csv("dataset/crop_recommendation.csv")

X = df.drop("label", axis=1)
y = df["label"]

# NO SCALING
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")

print(classification_report(y_test, y_pred))

pickle.dump(
    model,
    open("saved_models/random_forest_model.pkl", "wb")
)

print("\n✅ Random Forest Model Saved Successfully!")