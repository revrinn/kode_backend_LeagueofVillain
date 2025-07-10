from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.routes.predict import router as predict_router
from app.schemas.input_data import InputData
import joblib
import numpy as np
from typing import Dict, Any
import os

app = FastAPI(
    title="API Segmentasi Gaya Belajar",
    description="API untuk memprediksi segmentasi gaya belajar berdasarkan data aktivitas pembelajaran",
    version="1.0.0"
)

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables untuk model
model = None
scaler = None
features = None

@app.on_event("startup")
async def load_model():
    global model, scaler, features
    try:
        model = joblib.load("app/model/model.pkl")
        scaler = joblib.load("app/model/scaler.pkl")
        features = joblib.load("app/model/features.pkl")
        print("✅ Model berhasil dimuat")
    except Exception as e:
        print(f"❌ Gagal memuat model: {str(e)}")
        raise e

@app.get("/")
async def health_check():
    return {"message": "API Segmentasi Gaya Belajar berjalan dengan baik"}

@app.post("/predict/")
async def predict_direct(data: InputData):
    try:
        efficiency = data.completion_rate / (data.duration_minutes + 1e-6)
        
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
        
        input_scaled = scaler.transform([input_values])
        cluster = int(model.predict(input_scaled)[0])
        
        gaya_belajar_map = {
            0: "Visual",
            1: "Auditori", 
            2: "Kinestetik"
        }
        
        return {
            "status": "success",
            "predicted_gaya_belajar": gaya_belajar_map.get(cluster, f"Cluster {cluster}"),
            "cluster": cluster,
            "input_data": data.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include router
app.include_router(predict_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)