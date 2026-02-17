import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ---------------- CREATE DATASET ----------------

data = {
    "hour": [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22],
    "vehicles": [10,20,50,80,40,30,60,70,90,100,120,130,110,80,60,40,20],
    "weather": ["Sunny","Sunny","Cloudy","Rainy","Sunny","Cloudy","Rainy",
                "Sunny","Cloudy","Rainy","Sunny","Cloudy","Rainy",
                "Sunny","Cloudy","Rainy","Sunny"],
    "congestion": ["Low","Low","Medium","High","Low","Medium","High",
                   "Medium","High","High","High","High","High",
                   "Medium","Medium","Low","Low"]
}

df = pd.DataFrame(data)

# ---------------- ENCODE WEATHER ----------------

weather_map = {"Sunny":0,"Cloudy":1,"Rainy":2}
df["weather"] = df["weather"].map(weather_map)

# ---------------- ENCODE TARGET ----------------

target_map = {"Low":0,"Medium":1,"High":2}
df["congestion"] = df["congestion"].map(target_map)

# ---------------- SPLIT ----------------

X = df[["hour","vehicles","weather"]]
y = df["congestion"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- SCALE ----------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- TRAIN MODEL ----------------

model = RandomForestClassifier()

model.fit(X_train_scaled, y_train)

# ---------------- ACCURACY ----------------

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# ---------------- SAVE FILES ----------------

pickle.dump(model, open("traffic_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model and scaler saved successfully")
