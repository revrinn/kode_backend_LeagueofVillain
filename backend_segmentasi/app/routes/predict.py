from fastapi import APIRouter, HTTPException
from app.schemas.input_data import InputData
from typing import Dict, Any
import joblib
import numpy as np

router = APIRouter()

# Load model dan scaler saat modul di-load
try:
    model = joblib.load("app/model/model.pkl")
    scaler = joblib.load("app/model/scaler.pkl")
    print("✅ Model dan scaler berhasil dimuat di predict.py")
except Exception as e:
    print(f"❌ Gagal memuat model di predict.py: {str(e)}")
    raise

@router.post("/predict/", response_model=Dict[str, Any])
async def predict(data: InputData):
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
        
        # Scale input dan lakukan prediksi
        input_scaled = scaler.transform([input_values])
        cluster = int(model.predict(input_scaled)[0])  # Perbaikan di sini: [0] bukan [6]
        
        # Mapping cluster ke gaya belajar
        gaya_belajar_map = {
            0: "Visual",
            1: "Auditori", 
            2: "Kinestetik"
        }
        
        return {
            "status": "success",
            "predicted_gaya_belajar": gaya_belajar_map.get(cluster, f"Cluster {cluster}"),
            "cluster": cluster,
            "input_data": data.dict(),
            "efficiency": efficiency
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat prediksi: {str(e)}"
        )