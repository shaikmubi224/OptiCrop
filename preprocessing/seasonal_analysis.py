import pandas as pd

# Load dataset
data = pd.read_csv("dataset/crop_recommendation.csv")

print("=" * 60)
print("SEASONAL CROP ANALYSIS")
print("=" * 60)

# Summer crops
summer = data[
    (data["temperature"] > 30) &
    (data["humidity"] > 50)
]["label"].unique()

print("\n🌞 Summer Crops")
print(summer)

# Winter crops
winter = data[
    (data["temperature"] < 20) &
    (data["humidity"] > 30)
]["label"].unique()

print("\n❄ Winter Crops")
print(winter)

# Rainy crops
rainy = data[
    (data["rainfall"] > 200) &
    (data["humidity"] > 50)
]["label"].unique()

print("\n🌧 Rainy Crops")
print(rainy)