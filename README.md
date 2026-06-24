# Smart Irrigation ML Predictor 🌿

A Flask web app that uses a **Random Forest ML model** to predict whether irrigation is needed based on manually entered field conditions. No CSV upload, no IoT — pure ML.

## Project Structure
```
smart_irrigation/
├── app.py              # Flask backend + ML inference
├── train_model.py      # Model training script (run once)
├── requirements.txt
├── model/
│   ├── irrigation_model.pkl   # Trained RandomForest model
│   └── scaler.pkl             # StandardScaler
└── templates/
    └── index.html             # Frontend UI
```

## Setup in VS Code

### Step 1 — Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Train the model (only once)
```bash
python train_model.py
```
This creates `model/irrigation_model.pkl` and `model/scaler.pkl`.

### Step 4 — Run the app
```bash
python app.py
```
Open browser: **http://localhost:5000**

---

## Features
- 🎚️ **Sliders** for Soil Moisture, Temperature, Humidity, Rainfall
- 🌾 **Crop Type** selector (Wheat / Rice / Maize / Vegetable)
- 📅 **Day of Week** selector
- 📊 **Probability bars** showing model confidence
- 💡 **Smart advice** based on input values
- ✅ No CSV file needed — all manual input

## Model Details
| Feature       | Value |
|---------------|-------|
| Algorithm     | Random Forest Classifier |
| Trees         | 150 |
| Max Depth     | 10 |
| Training Size | 1600 samples |
| Test Accuracy | **88.25%** |

## Input Features
| Feature       | Range   | Description |
|---------------|---------|-------------|
| Soil Moisture | 0–100%  | Current soil water content |
| Temperature   | 0–60°C  | Ambient air temperature |
| Humidity      | 0–100%  | Relative humidity |
| Rainfall      | 0–300mm | Last 24h precipitation |
| Crop Type     | 4 types | Wheat / Rice / Maize / Vegetable |
| Day of Week   | Mon–Sun | Current day |
