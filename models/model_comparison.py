import pandas as pd

results = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "KNN"
    ],
    "Accuracy": [
        96.36,
        98.64,
        99.32,
        95.68
    ]
}

df = pd.DataFrame(results)

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(df)

best = df.loc[df["Accuracy"].idxmax()]

print("\nBest Model")
print(f"Model    : {best['Model']}")
print(f"Accuracy : {best['Accuracy']}%")

df.to_csv("models/model_comparison.csv", index=False)

print("\n✅ Model Comparison Report Saved Successfully!")