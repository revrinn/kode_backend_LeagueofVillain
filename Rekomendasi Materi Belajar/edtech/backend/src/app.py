# backend/src/app.py/recommendation
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from typing import List, Optional
import uvicorn
from pathlib import Path
from recommendation.collaborative import CollaborativeFiltering
from recommendation.content_based import ContentBasedRecommender
from recommendation.hybrid import HybridRecommender

# ===== KONFIGURASI SERVER =====
HOST = "0.0.0.0"  #untuk deploy hugging face
PORT = 8025
RELOAD = True  # Set False di production
WORKERS = 1

# ===== LIFESPAN MANAGEMENT =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Mengelola siklus hidup aplikasi dan inisialisasi model"""
    print("Memuat model rekomendasi...")
    try:
        # Load semua model
        app.state.collab_model = CollaborativeFiltering.load_model(COLLAB_MODEL_PATH)
        app.state.content_model = ContentBasedRecommender.load_model(CONTENT_MODEL_PATH)
        app.state.hybrid_model = HybridRecommender.load_model(
            collab_path=COLLAB_MODEL_PATH,
            content_path=CONTENT_MODEL_PATH,
            hybrid_path=HYBRID_MODEL_PATH
        )
        print("✅ Model berhasil dimuat!")
    except Exception as e:
        print(f"❌ Gagal memuat model: {str(e)}")
        raise HTTPException(status_code=500, detail="Gagal memuat model")
    yield
    print("🛑 Server dimatikan")

# ===== INISIALISASI APLIKASI =====
app = FastAPI(
    title="Sistem Rekomendasi Materi Pembelajaran",
    description="API untuk memberikan rekomendasi materi pembelajaran personalisasi",
    version="1.0.2",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ===== KONFIGURASI CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3025"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== PATH MODEL =====
MODEL_DIR = Path("models/recommenders")
COLLAB_MODEL_PATH = MODEL_DIR / "collaborative/collab_model.joblib"
CONTENT_MODEL_PATH = MODEL_DIR / "content_based/content_model.joblib"
HYBRID_MODEL_PATH = MODEL_DIR / "hybrid/hybrid_model.joblib"

# ===== SCHEMA REQUEST/RESPONSE =====
class RecommendationRequest(BaseModel):
    user_id: str
    user_history: List[str]
    n_recommendations: int = 5
    algorithm: str = "hybrid"

class MaterialRecommendationRequest(BaseModel):
    material_id: str
    n_recommendations: int = 5

class RecommendationItem(BaseModel):
    material_id: str
    score: float
    confidence: float = 0.0  # Tambahan field baru

class RecommendationResponse(BaseModel):
    success: bool
    recommendations: List[RecommendationItem]
    algorithm: str
    message: Optional[str] = None

# ===== ENDPOINT API =====
@app.get("/")
async def root():
    return {
        "message": "Selamat datang di API Rekomendasi Pembelajaran",
        "version": app.version,
        "docs": f"http://{HOST}:{PORT}/docs"
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        # Validasi input
        if not request.user_id:
            raise HTTPException(
                status_code=400,
                detail="User ID diperlukan",
                headers={"Content-Type": "application/json"}
            )
            
        if not request.user_history and request.algorithm != "collaborative":
            raise HTTPException(
                status_code=400,
                detail="User history diperlukan untuk algoritma ini",
                headers={"Content-Type": "application/json"}
            )
        # Format response yang lebih konsisten
        recommendations = []
        if request.algorithm == "hybrid":
            recommendations = app.state.hybrid_model.recommend_for_user(
                user_id=request.user_id,
                user_history=request.user_history or [],  # Handle None
                df=pd.DataFrame(),
                n_recommendations=request.n_recommendations
            )
        elif request.algorithm == "collaborative":
            recommendations = app.state.collab_model.recommend_for_user(
                user_id=request.user_id
            )[:request.n_recommendations]
        else:
            recommendations = app.state.content_model.recommend_for_user(
                user_id=request.user_id,
                user_history=request.user_history or [],  # Handle None
                df=pd.DataFrame()
            )[:request.n_recommendations]

        # Pastikan format response konsisten
        recommendation_items = [
            {
                "material_id": item[0], 
                "score": float(item[1]),
                "confidence": min(float(item[1]) * 100, 99.9)
            }
            for item in recommendations
        ]
        
        return {
            "success": True,
            "recommendations": recommendation_items,
            "algorithm": request.algorithm,
            "message": "Rekomendasi berhasil dibuat"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
            headers={"Content-Type": "application/json"}
        )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if all([
            hasattr(app.state, "collab_model"),
            hasattr(app.state, "content_model"),
            hasattr(app.state, "hybrid_model")
        ]) else "unhealthy",
        "details": {
            "collaborative_loaded": hasattr(app.state, "collab_model"),
            "content_loaded": hasattr(app.state, "content_model"),
            "hybrid_loaded": hasattr(app.state, "hybrid_model")
        }
    }

# ===== KONFIGURASI SERVER =====
def run_server():
    """Menjalankan server Uvicorn"""
    config = uvicorn.Config(
        app,
        host=HOST,
        port=PORT,
        reload=RELOAD,
        workers=WORKERS,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    print(f"🚀 Server berjalan di http://{HOST}:{PORT}")
    print(f"📚 Dokumentasi API tersedia di http://{HOST}:{PORT}/docs")
    
    server.run()

if __name__ == "__main__":
    run_server()