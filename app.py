from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE = os.path.dirname(__file__)
model  = joblib.load(os.path.join(BASE, "model", "irrigation_model.pkl"))
scaler = joblib.load(os.path.join(BASE, "model", "scaler.pkl"))

CROP_MAP = {"wheat": 0, "rice": 1, "maize": 2, "vegetable": 3}
DAY_MAP   = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,
             "friday":4,"saturday":5,"sunday":6}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        soil_moisture = float(data["soil_moisture"])
        temperature   = float(data["temperature"])
        humidity      = float(data["humidity"])
        rainfall      = float(data["rainfall"])
        crop_type     = int(CROP_MAP.get(data["crop_type"].lower(), 0))
        day_of_week   = int(DAY_MAP.get(data["day_of_week"].lower(), 0))

        errors = []
        if not (0 <= soil_moisture <= 100):  errors.append("Soil moisture must be 0-100%")
        if not (0 <= temperature   <= 60):   errors.append("Temperature must be 0-60 C")
        if not (0 <= humidity      <= 100):  errors.append("Humidity must be 0-100%")
        if not (0 <= rainfall      <= 300):  errors.append("Rainfall must be 0-300 mm")
        if errors:
            return jsonify({"error": "; ".join(errors)}), 400

        X = np.array([[soil_moisture, temperature, humidity, rainfall, crop_type, day_of_week]])
        X_sc = scaler.transform(X)
        pred  = model.predict(X_sc)[0]
        proba = model.predict_proba(X_sc)[0]

        advice = []
        if soil_moisture < 30:
            advice.append("Soil is very dry — immediate watering needed.")
        elif soil_moisture > 70:
            advice.append("Soil moisture is adequate — no watering required.")
        if temperature > 38:
            advice.append("High temperature — consider evening irrigation.")
        if rainfall > 20:
            advice.append("Heavy rainfall — hold irrigation to avoid waterlogging.")
        if humidity < 30:
            advice.append("Low humidity — increase irrigation frequency.")

        return jsonify({
            "prediction":    int(pred),
            "label":         "Irrigate Now" if pred == 1 else "No Irrigation Needed",
            "confidence":    round(float(max(proba)) * 100, 1),
            "irrigate_prob": round(float(proba[1]) * 100, 1),
            "no_irr_prob":   round(float(proba[0]) * 100, 1),
            "advice":        advice
        })

    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "RandomForestClassifier", "accuracy": "88.25%"})

if __name__ == "__main__":
    import os
app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
