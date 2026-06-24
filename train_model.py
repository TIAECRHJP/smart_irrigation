"""
Smart Irrigation ML Model Trainer
Generates synthetic training data and trains a Random Forest Classifier.
Run this once to create the model file.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

np.random.seed(42)
N = 2000

# Features: soil_moisture(%), temperature(°C), humidity(%), rainfall(mm), crop_type(0-3), day_of_week(0-6)
soil_moisture   = np.random.uniform(10, 90, N)
temperature     = np.random.uniform(15, 45, N)
humidity        = np.random.uniform(20, 95, N)
rainfall        = np.random.uniform(0, 50, N)
crop_type       = np.random.randint(0, 4, N)   # 0=wheat,1=rice,2=maize,3=vegetable
day_of_week     = np.random.randint(0, 7, N)

# Rule-based label: irrigate=1 when soil is dry, hot, low rain, low humidity
def label(sm, temp, hum, rain, crop):
    score = 0
    if sm < 35:   score += 3
    elif sm < 50: score += 1
    if temp > 35: score += 2
    elif temp > 28: score += 1
    if hum < 40:  score += 1
    if rain < 5:  score += 2
    elif rain < 15: score += 1
    # Rice needs more water
    if crop == 1 and sm < 60: score += 2
    # Vegetables are sensitive
    if crop == 3 and sm < 45: score += 1
    return 1 if score >= 4 else 0

labels = np.array([label(soil_moisture[i], temperature[i], humidity[i],
                          rainfall[i], crop_type[i]) for i in range(N)])

# Add small noise
noise_idx = np.random.choice(N, size=int(N * 0.05), replace=False)
labels[noise_idx] = 1 - labels[noise_idx]

X = np.column_stack([soil_moisture, temperature, humidity, rainfall, crop_type, day_of_week])
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["No Irrigation", "Irrigate"]))

os.makedirs("model", exist_ok=True)
joblib.dump(model,  "model/irrigation_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("Model saved to model/")
