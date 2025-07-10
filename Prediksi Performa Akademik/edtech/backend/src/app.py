from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import pandas as pd
import joblib
from pathlib import Path
import numpy as np
import sys
import logging
import time
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
import xgboost as xgb
import shap
import json
from contextlib import asynccontextmanager
from datetime import datetime
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api.log')
    ]
)
logger = logging.getLogger(__name__)

# Setup paths - Disesuaikan dengan struktur folder Anda
BASE_DIR = Path(__file__).parent.parent  # Menyesuaikan dengan lokasi app.py
MODEL_DIR = BASE_DIR / "models" / "performance_predictor" / "trained_model"
MODEL_PATH = MODEL_DIR / "performance_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "data_processor.pkl"
METRICS_PATH = MODEL_DIR / "model_metrics.json"

# Pastikan direktori model ada
os.makedirs(MODEL_DIR, exist_ok=True)

# Lifespan handler untuk manajemen siklus hidup aplikasi
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Mengelola startup dan shutdown aplikasi"""
    try:
        # Muat komponen saat startup
        app.state.model_components = await load_components()
        
        # Muat metrik model
        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                app.state.model_metrics = json.load(f)
        else:
            app.state.model_metrics = {
                "mse": 0.05486344948816889,
                "rmse": 0.23422948039939143,
                "mae": 0.1660625786187038,
                "r2": 0.29007536468986816,
                "max_error": 0.7487417459487915
            }
            logger.warning("File metrik model tidak ditemukan, menggunakan nilai default")
        
        logger.info("Aplikasi siap menerima request")
        yield
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal memulai aplikasi"
        )

# Inisialisasi FastAPI
app = FastAPI(
    title="EdTech Performance Prediction API",
    description="API untuk memprediksi performa akademik siswa menggunakan model XGBoost",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3024", "http://192.168.56.1:3024"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Definisi Model Pydantic
class FeatureInput(BaseModel):
    grade: float = Field(..., gt=0, le=12, description="Kelas siswa (1-12)")
    tech_savvy: int = Field(..., ge=1, le=5, description="Kemampuan teknologi (skala 1-5)")
    duration_minutes: float = Field(..., gt=0, description="Durasi belajar dalam menit")
    engagement_score: float = Field(..., ge=0, le=1, description="Skor engagement (0-1)")
    completion_rate: float = Field(..., ge=0, le=1, description="Tingkat penyelesaian materi (0-1)")
    material_rating: float = Field(..., ge=1, le=5, description="Rating materi (skala 1-5)")
    interaction_duration: float = Field(..., gt=0, description="Durasi interaksi dengan materi")
    material_engagement_score: float = Field(..., ge=0, le=1, description="Skor engagement dengan materi")
    feature_engagement: float = Field(..., ge=0, le=1, description="Engagement dengan fitur platform")
    jam_belajar: float = Field(..., ge=0, le=24, description="Jam belajar (0-24)")
    hari_dalam_minggu: float = Field(..., ge=0, le=6, description="Hari dalam minggu (0-6)")
    akhir_pekan: float = Field(..., ge=0, le=1, description="Indikator akhir pekan (0/1)")
    efisiensi_belajar: float = Field(..., ge=0, description="Indeks efisiensi belajar")
    rasio_penyelesaian: float = Field(..., ge=0, le=1, description="Rasio penyelesaian tugas")
    interaksi_total: float = Field(..., ge=0, description="Total interaksi dengan platform")
    preferensi_materi: float = Field(..., ge=0, le=1, description="Preferensi jenis materi")
    jumlah_pengakses: float = Field(..., ge=0, description="Jumlah pengakses materi")
    engagement_rata2: float = Field(..., ge=0, le=1, description="Rata-rata engagement")
    performance_label_encoded: int = Field(..., ge=0, description="Label performa (encoded)")
    learning_speed_encoded: int = Field(..., ge=0, description="Kecepatan belajar (encoded)")
    student_feedback_encoded: int = Field(..., ge=0, description="Feedback siswa (encoded)")
    achievement_status_encoded: int = Field(..., ge=0, description="Status pencapaian (encoded)")

    @field_validator('engagement_score', 'completion_rate', 'material_engagement_score', 
                   'feature_engagement', 'efisiensi_belajar', 'rasio_penyelesaian',
                   'preferensi_materi', 'engagement_rata2')
    @classmethod
    def check_proportion(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Nilai harus antara 0 dan 1")
        return v

class PredictionInput(BaseModel):
    features: FeatureInput

class BatchPredictionInput(BaseModel):
    samples: List[FeatureInput]

class FeatureContribution(BaseModel):
    feature: str
    value: float
    contribution: float

class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Nilai prediksi skor kuis")
    confidence_interval: List[float] = Field(..., description="Interval kepercayaan prediksi")
    feature_contributions: Optional[List[FeatureContribution]] = Field(
        None, 
        description="Kontribusi masing-masing fitur terhadap prediksi"
    )
    execution_time_ms: float = Field(..., description="Waktu eksekusi dalam milidetik")
    model_version: str = Field(..., description="Versi model yang digunakan")

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    confidence_intervals: List[List[float]]
    feature_contributions: Optional[List[List[FeatureContribution]]]
    execution_time_ms: float
    model_version: str
    total_samples: int
    avg_time_per_sample_ms: float

class HealthCheckResponse(BaseModel):
    status: str
    model_version: str
    model_metrics: dict
    uptime_seconds: float

class ModelInfoResponse(BaseModel):
    features: List[str]
    model_type: str
    training_date: Optional[str]
    performance_metrics: dict

# Dependency untuk memuat komponen model
async def load_components():
    """Memuat model dan preprocessor dari file"""
    try:
        start_time = time.time()
        
        # Verifikasi file ada
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"File model tidak ditemukan di {MODEL_PATH}")
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"File preprocessor tidak ditemukan di {PREPROCESSOR_PATH}")
        
        # Load model
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model berhasil dimuat dari {MODEL_PATH}")
        
        # Load preprocessor
        processor_data = joblib.load(PREPROCESSOR_PATH)
        preprocessor = processor_data['preprocessor']
        feature_names = processor_data['feature_names']
        logger.info(f"Preprocessor berhasil dimuat dari {PREPROCESSOR_PATH}")
        
        load_time = time.time() - start_time
        logger.info(f"Komponen model berhasil dimuat dalam {load_time:.2f} detik")
        
        return {
            "model": model,
            "preprocessor": preprocessor,
            "feature_names": feature_names,
            "load_time": load_time
        }
    except FileNotFoundError as e:
        logger.error(f"File tidak ditemukan: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File model/preprocessor tidak ditemukan: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Gagal memuat model/preprocessor: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gagal memuat komponen model: {str(e)}"
        )

# Endpoint Utama
@app.get("/", include_in_schema=False)
async def root():
    """Endpoint root untuk informasi dasar API"""
    return {
        "message": "Selamat datang di EdTech Performance Prediction API",
        "version": app.version,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict",
            "batch_predict": "/predict/batch"
        }
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Endpoint untuk health check dan monitoring"""
    return {
        "status": "healthy",
        "model_version": app.version,
        "model_metrics": app.state.model_metrics,
        "uptime_seconds": time.time() - app.state.model_components.get("load_time", time.time())
    }

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Endpoint untuk mendapatkan informasi tentang model"""
    return {
        "features": app.state.model_components["feature_names"],
        "model_type": "XGBoost Regressor",
        "training_date": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat(),
        "performance_metrics": app.state.model_metrics
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_performance(
    input_data: PredictionInput
):
    """Endpoint untuk prediksi tunggal performa siswa"""
    start_time = time.time()
    
    try:
        components = app.state.model_components
        model = components["model"]
        preprocessor = components["preprocessor"]
        feature_names = components["feature_names"]
        
        # Konversi input ke DataFrame
        input_dict = input_data.features.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Validasi fitur
        missing_cols = set(feature_names) - set(input_df.columns)
        if missing_cols:
            raise ValueError(f"Kolom berikut tidak ditemukan: {missing_cols}")
        
        # Urutkan kolom sesuai dengan yang diharapkan model
        input_df = input_df[feature_names]
        
        # Preprocess input
        processed_input = preprocessor.transform(input_df)
        
        # Buat prediksi
        if isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix(processed_input)
            prediction = model.predict(dmatrix)[0]
        else:
            prediction = model.predict(processed_input)[0]
        
        # Hitung confidence interval berdasarkan metrik model
        std_dev = np.sqrt(app.state.model_metrics.get('mse', 0.05486344948816889))
        confidence = [max(0, prediction - 1.96*std_dev), min(1, prediction + 1.96*std_dev)]
        
        # Hitung feature contributions menggunakan SHAP
        feature_contributions = None
        if hasattr(model, 'feature_names_in_'):
            try:
                explainer = shap.Explainer(model)
                shap_values = explainer(processed_input)
                
                feature_contributions = []
                for i, feature in enumerate(feature_names):
                    feature_contributions.append({
                        "feature": feature,
                        "value": input_df.iloc[0][feature],
                        "contribution": float(shap_values[0].values[i])
                    })
                # Urutkan berdasarkan kontribusi absolut terbesar
                feature_contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
            except Exception as e:
                logger.warning(f"Tidak dapat menghitung SHAP values: {str(e)}")
        
        # Hitung waktu response
        exec_time = (time.time() - start_time) * 1000  # dalam milidetik
        
        return {
            "prediction": float(prediction),
            "confidence_interval": confidence,
            "feature_contributions": feature_contributions,
            "execution_time_ms": exec_time,
            "model_version": app.version
        }
        
    except ValueError as e:
        logger.error(f"Input validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Input tidak valid: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error dalam prediksi: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error dalam prediksi: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict_performance(
    input_data: BatchPredictionInput
):
    """Endpoint untuk prediksi batch performa siswa"""
    start_time = time.time()
    
    try:
        components = app.state.model_components
        model = components["model"]
        preprocessor = components["preprocessor"]
        feature_names = components["feature_names"]
        
        # Konversi input ke DataFrame
        samples = [sample.dict() for sample in input_data.samples]
        input_df = pd.DataFrame(samples)
        
        # Validasi fitur
        missing_cols = set(feature_names) - set(input_df.columns)
        if missing_cols:
            raise ValueError(f"Kolom berikut tidak ditemukan: {missing_cols}")
        
        # Urutkan kolom
        input_df = input_df[feature_names]
        
        # Preprocess input
        processed_input = preprocessor.transform(input_df)
        
        # Buat prediksi
        if isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix(processed_input)
            predictions = model.predict(dmatrix)
        else:
            predictions = model.predict(processed_input)
        
        # Hitung confidence intervals
        std_dev = np.sqrt(app.state.model_metrics.get('mse', 0.05486344948816889))
        conf_intervals = [
            [max(0, p - 1.96*std_dev), min(1, p + 1.96*std_dev)] 
            for p in predictions
        ]
        
        # Hitung feature contributions
        feature_contributions_list = None
        if hasattr(model, 'feature_names_in_'):
            try:
                explainer = shap.Explainer(model)
                shap_values = explainer(processed_input)
                
                feature_contributions_list = []
                for i in range(len(predictions)):
                    contributions = []
                    for j, feature in enumerate(feature_names):
                        contributions.append({
                            "feature": feature,
                            "value": input_df.iloc[i][feature],
                            "contribution": float(shap_values[i].values[j])
                        })
                    # Urutkan berdasarkan kontribusi absolut terbesar
                    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
                    feature_contributions_list.append(contributions)
            except Exception as e:
                logger.warning(f"Tidak dapat menghitung SHAP values untuk batch: {str(e)}")
        
        # Hitung waktu response
        exec_time = (time.time() - start_time) * 1000  # dalam milidetik
        avg_time_per_sample = exec_time / len(predictions)
        
        return {
            "predictions": [float(p) for p in predictions],
            "confidence_intervals": conf_intervals,
            "feature_contributions": feature_contributions_list,
            "execution_time_ms": exec_time,
            "model_version": app.version,
            "total_samples": len(predictions),
            "avg_time_per_sample_ms": avg_time_per_sample
        }
        
    except ValueError as e:
        logger.error(f"Input validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Input tidak valid: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error dalam batch prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error dalam batch prediction: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="192.168.56.1",
        port=8024,
        reload=True
    )