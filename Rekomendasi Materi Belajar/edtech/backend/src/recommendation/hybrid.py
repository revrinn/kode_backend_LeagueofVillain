# backend/src/recommendation/hybrid.py
from .collaborative import CollaborativeFiltering
from .content_based import ContentBasedRecommender
import numpy as np
import joblib
from pathlib import Path

class HybridRecommender:
    def __init__(self, collab_model, content_model, alpha=0.5):
        self.collab_model = collab_model
        self.content_model = content_model
        self.alpha = alpha
    
    def recommend_for_user(self, user_id, user_history, df, n_recommendations=5):
        """
        Memberikan rekomendasi hybrid untuk user tertentu
        
        Parameters:
        - user_id: ID user (wajib)
        - user_history: List material_id yang pernah diakses user
        - df: DataFrame lengkap data materi
        - n_recommendations: Jumlah rekomendasi
        """
        if not user_id or not user_history or len(user_history) < 1:
            return []
            
        try:
            # Dapatkan rekomendasi collaborative
            collab_recs = self.collab_model.recommend_for_user(user_id) or []
            
            # Dapatkan rekomendasi content-based
            content_recs = self.content_model.recommend_for_user(
                user_id=user_id,
                user_history=user_history,
                df=df
            ) or []
            
            # Jika salah satu kosong, gunakan yang lain
            if not collab_recs and not content_recs:
                return []
            elif not collab_recs:
                return content_recs[:n_recommendations]
            elif not content_recs:
                return collab_recs[:n_recommendations]
                
            # Gabungkan rekomendasi
            hybrid_scores = self._combine_recommendations(collab_recs, content_recs)
            hybrid_scores = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
            
            return hybrid_scores[:n_recommendations]
        except Exception as e:
            print(f"Error pada hybrid recommender untuk user {user_id}: {str(e)}")
            return []
    
    def _combine_recommendations(self, collab_recs, content_recs):
        """
        Menggabungkan skor dari kedua model dengan normalisasi yang lebih baik
        """
        # Normalisasi skor collaborative
        collab_scores = {item: score for item, score in collab_recs}
        if collab_scores:
            max_collab = max(collab_scores.values()) if max(collab_scores.values()) != 0 else 1
            min_collab = min(collab_scores.values())
            collab_scores = {k: (v - min_collab)/(max_collab - min_collab + 1e-10) 
                            for k, v in collab_scores.items()}
        
        # Normalisasi skor content-based
        content_scores = {item: score for item, score in content_recs}
        if content_scores:
            max_content = max(content_scores.values()) if max(content_scores.values()) != 0 else 1
            min_content = min(content_scores.values())
            content_scores = {k: (v - min_content)/(max_content - min_content + 1e-10)
                            for k, v in content_scores.items()}
        
        # Gabungkan semua material yang direkomendasikan
        all_items = set(collab_scores.keys()).union(set(content_scores.keys()))
        
        # Hitung hybrid score dengan penyesuaian dinamis
        hybrid_scores = {}
        for item in all_items:
            collab_score = collab_scores.get(item, 0)
            content_score = content_scores.get(item, 0)
            
            # Adjust alpha based on score confidence
            effective_alpha = self.alpha
            if len(collab_recs) < 3:  # Jika terlalu sedikit rekomendasi collab
                effective_alpha = 0.3
            
            hybrid_score = (effective_alpha * collab_score) + ((1 - effective_alpha) * content_score)
            hybrid_scores[item] = hybrid_score
        
        return hybrid_scores

    
    def save_model(self, save_path='models/recommenders/hybrid'):
        """
        Menyimpan model hybrid (sebenarnya menyimpan referensi ke model lain)
        """
        # Tidak perlu menyimpan model hybrid karena hanya kombinasi dari model lain
        # Tetapi kita bisa menyimpan parameter alpha
        model_data = {
            'alpha': self.alpha
        }
        
        Path(save_path).mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, f'{save_path}/hybrid_model.joblib')
        print("Parameter Hybrid Recommender berhasil disimpan!")
    
    @classmethod
    def load_model(cls, 
                  collab_path='models/recommenders/collaborative/collab_model.joblib',
                  content_path='models/recommenders/content_based/content_model.joblib',
                  hybrid_path='models/recommenders/hybrid/hybrid_model.joblib'):
        """
        Memuat model hybrid dengan memuat model dasar terlebih dahulu
        """
        # Muat model collaborative dan content-based
        collab_model = CollaborativeFiltering.load_model(collab_path)
        content_model = ContentBasedRecommender.load_model(content_path)
        
        # Muat parameter hybrid
        hybrid_data = joblib.load(hybrid_path)
        
        # Buat instance hybrid recommender
        model = cls(collab_model, content_model, alpha=hybrid_data['alpha'])
        
        return model