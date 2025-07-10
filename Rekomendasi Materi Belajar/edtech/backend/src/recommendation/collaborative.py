# backend/src/recommendation/collaborative.py
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from pathlib import Path
from scipy.sparse import csr_matrix

class CollaborativeFiltering:
    def __init__(self, n_factors=50, n_recommendations=5):
        self.n_factors = n_factors
        self.n_recommendations = n_recommendations
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.user_ids = None
        self.item_ids = None
    
    def fit(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.user_ids = user_item_matrix.index
        self.item_ids = user_item_matrix.columns
        
        # Normalisasi dengan subtract mean
        user_means = user_item_matrix.mean(axis=1)
        normalized_matrix = user_item_matrix.sub(user_means, axis=0).fillna(0)
        
        # Convert the matrix to sparse format (CSR format)
        sparse_matrix = csr_matrix(normalized_matrix.values)
        
        # Tentukan nilai k secara dinamis untuk dataset kecil
        min_dim = min(sparse_matrix.shape)
        k = min(self.n_factors, min_dim - 1) if min_dim > 1 else 1
        
        # Jika dimensi terlalu kecil, gunakan similarity dasar
        if k < 1:
            print("Matriks terlalu kecil, menggunakan similarity dasar")
            self.similarity_matrix = cosine_similarity(normalized_matrix.T)
            return
        
        print(f"Menentukan k = {k} berdasarkan dimensi matriks: {sparse_matrix.shape}")
        
        try:
            # Melakukan SVD dengan penanganan khusus untuk matriks kecil
            U, sigma, Vt = svds(sparse_matrix, k=k)
            
            # Mengubah sigma menjadi matriks diagonal
            sigma = np.diag(sigma)
            
            # Membuat user dan item factors
            self.user_factors = U
            self.item_factors = sigma @ Vt
        except Exception as e:
            print(f"Error dalam SVD: {str(e)} - menggunakan similarity dasar")
            self.similarity_matrix = cosine_similarity(normalized_matrix.T)

    def recommend_for_user(self, user_id, user_item_matrix=None):
        if user_item_matrix is not None:
            self.user_item_matrix = user_item_matrix
        
        # Handle jika user_id tidak ada di data training
        if user_id not in self.user_ids:
            print(f"User ID {user_id} tidak ditemukan di model")
            # Fallback: return popular items
            item_counts = (self.user_item_matrix > 0).sum()
            top_items = item_counts.sort_values(ascending=False).head(self.n_recommendations).index
            return [(item, 0.5) for item in top_items]
        
        try:
            # Jika menggunakan similarity dasar
            if hasattr(self, 'similarity_matrix'):
                user_idx = np.where(self.user_ids == user_id)[0][0]
                user_ratings = self.user_item_matrix.iloc[user_idx].values
                unseen_mask = user_ratings == 0
                item_scores = self.similarity_matrix.dot(user_ratings)
                item_scores[~unseen_mask] = -np.inf  # Filter yang sudah dilihat
                top_indices = np.argsort(-item_scores)[:self.n_recommendations]
                return [(self.item_ids[i], item_scores[i]) for i in top_indices if item_scores[i] > 0]
            
            # Jika menggunakan SVD
            user_idx = np.where(self.user_ids == user_id)[0][0]
            user_ratings = self.user_factors[user_idx, :] @ self.item_factors
            
            # Dapatkan item yang belum dilihat user
            known_items = self.user_item_matrix.loc[user_id]
            unseen_items_idx = np.where(known_items == 0)[0]
            
            # Jika tidak ada item yang belum dilihat, kembalikan popular items
            if len(unseen_items_idx) == 0:
                item_counts = (self.user_item_matrix > 0).sum()
                top_items = item_counts.sort_values(ascending=False).head(self.n_recommendations).index
                return [(item, 0.5) for item in top_items]
            
            # Urutkan item yang belum dilihat berdasarkan prediksi rating
            unseen_ratings = user_ratings[unseen_items_idx]
            recommended_idx = np.argsort(-unseen_ratings)[:self.n_recommendations]
            
            # Buat rekomendasi
            recommendations = []
            for idx in recommended_idx:
                item_id = self.item_ids[unseen_items_idx[idx]]
                score = unseen_ratings[idx]
                recommendations.append((item_id, score))
            
            return recommendations
        except Exception as e:
            print(f"Error dalam rekomendasi untuk user {user_id}: {str(e)}")
            # Fallback: return popular items
            item_counts = (self.user_item_matrix > 0).sum()
            top_items = item_counts.sort_values(ascending=False).head(self.n_recommendations).index
            return [(item, 0.5) for item in top_items]
        
    def save_model(self, save_path='models/recommenders/collaborative'):
        """
        Menyimpan model yang sudah dilatih
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_ids': self.user_ids,
            'item_ids': self.item_ids,
            'n_factors': self.n_factors
        }
        
        joblib.dump(model_data, f'{save_path}/collab_model.joblib')
        print("Model Collaborative Filtering berhasil disimpan!")
    
    @classmethod
    def load_model(cls, load_path='models/recommenders/collaborative/collab_model.joblib'):
        """
        Memuat model yang sudah disimpan
        """
        model_data = joblib.load(load_path)
        
        model = cls(n_factors=model_data['n_factors'])
        model.user_factors = model_data['user_factors']
        model.item_factors = model_data['item_factors']
        model.user_ids = model_data['user_ids']
        model.item_ids = model_data['item_ids']
        
        return model