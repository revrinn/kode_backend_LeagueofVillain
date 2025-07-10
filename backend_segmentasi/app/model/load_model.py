import joblib
import numpy as np
from app.schemas.input_data import InputData
import os

# Cek keberadaan file model
model_path = "app/model/model.pkl"
scaler_path = "app/model/scaler.pkl"
features_path = "app/model/features.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
if not os.path.exists(features_path):
    raise FileNotFoundError(f"Features file not found: {features_path}")

# Load model, scaler, and feature list
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)
    print("✅ Model berhasil dimuat dari load_model.py")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise e

def predict_style(data: InputData):
    try:
        # Hitung feature tambahan 'efficiency'
        efficiency = data.completion_rate / (data.duration_minutes + 1e-6)  # Hindari division by zero
        
        input_values = [
            data.engagement_score,
            data.duration_minutes,
            data.completion_rate,
            data.quiz_score,
            data.material_rating,
            data.interaction_duration,
            data.material_engagement_score,
            efficiency
        ]
        
        # Scale input
        input_scaled = scaler.transform([input_values])
        
        # Predict cluster
        cluster = model.predict(input_scaled)[0]
        return int(cluster)
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")