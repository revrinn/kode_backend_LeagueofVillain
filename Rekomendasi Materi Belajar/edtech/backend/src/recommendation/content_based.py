# backend/src/recommendation/content_based.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

class ContentBasedRecommender:
    def __init__(self, n_recommendations=5):
        self.n_recommendations = n_recommendations
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.material_features = None
        self.material_ids = None
    
    def fit(self, df):
        # Gabungkan fitur teks materi dengan lebih banyak fitur untuk dataset kecil
        df['material_features'] = (
            df['related_materials'].fillna('') + " " + 
            df['subject_English'].astype(str) + " " +
            df['subject_History'].astype(str) + " " +
            df['subject_Mathematics'].astype(str) + " " +
            df['subject_Science'].astype(str) + " " +
            df['material_type_encoded'].astype(str) + " " +
            df['preferensi_materi'].fillna('').astype(str) + " " +
            df['performance_label_encoded'].astype(str)
        )

        # Simpan mapping material_id untuk referensi
        self.material_ids = df['material_type_encoded'].unique()
        
        # Inisialisasi TF-IDF Vectorizer dengan parameter untuk data kecil
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,  # Term muncul di minimal 1 dokumen
            max_df=0.95,  # Term muncul di maksimal 95% dokumen
            max_features=1000  # Batasi jumlah fitur
        )
        
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['material_features'])
            self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        except Exception as e:
            print(f"Error dalam TF-IDF: {str(e)}")
            # Buat matriks identitas sebagai fallback
            n = len(df)
            self.cosine_sim = np.eye(n)
        
        # Buat mapping antara index dan material_id dengan fallback
        self.indices = pd.Series(df.index, index=df['material_type_encoded']).drop_duplicates()

    def recommend_for_user(self, user_id, user_history, df):
        """Rekomendasi untuk user berdasarkan riwayat"""
        if not user_history or len(user_history) < 1:
            # Return default recommendations with adjusted scores
            top_materials = df['material_type_encoded'].value_counts().head(self.n_recommendations).index.tolist()
            return [(mat, 0.5 * df[df['material_type_encoded'] == mat]['engagement_score'].mean()) 
                    for mat in top_materials]
        
        try:
            # Dapatkan materi yang pernah diakses user
            user_materials = df[df['material_type_encoded'].isin(user_history)]
            if len(user_materials) == 0:
                return []
                
            # Hitung profil user dengan normalisasi
            user_profile = self._create_user_profile(user_history, df)
            if user_profile is None:
                return []
            
            # Hitung similarity dengan normalisasi
            user_profile = user_profile.reshape(1, -1)
            cosine_sim = linear_kernel(user_profile, self.tfidf_matrix)
            cosine_sim = (cosine_sim - cosine_sim.min()) / (cosine_sim.max() - cosine_sim.min() + 1e-10)
            
            # Gabungkan dengan engagement score
            material_scores = {}
            for idx, score in enumerate(cosine_sim[0]):
                material_id = df.iloc[idx]['material_type_encoded']
                if material_id not in user_history:
                    engagement = df[df['material_type_encoded'] == material_id]['engagement_score'].mean()
                    material_scores[material_id] = 0.7 * score + 0.3 * (engagement / 5.0)  # Normalisasi
            
            # Urutkan dan kembalikan rekomendasi
            recommendations = sorted(material_scores.items(), key=lambda x: x[1], reverse=True)
            return recommendations[:self.n_recommendations]
            
        except Exception as e:
            print(f"Error generating recommendations for user {user_id}: {str(e)}")
            return []
        
    def recommend_for_material(self, material_id):
        """
        Memberikan rekomendasi berdasarkan similarity konten
        
        Parameters:
        - material_id: ID materi yang akan dicari similaritasnya
        
        Returns:
        - recommendations: List rekomendasi material beserta similarity scores
        """
        try:
            idx = self.indices[material_id]
        except KeyError:
            print(f"Material ID {material_id} tidak ditemukan")
            return []
        
        # Dapatkan similarity scores untuk semua materi
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Urutkan berdasarkan similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Ambil n_recommendations teratas (tidak termasuk diri sendiri)
        sim_scores = sim_scores[1:self.n_recommendations+1]
        
        # Dapatkan material indices
        material_indices = [i[0] for i in sim_scores]
        
        # Buat rekomendasi
        recommendations = []
        for i, (idx, score) in enumerate(sim_scores):
            rec_material_id = self.material_ids[material_indices[i]]
            recommendations.append((rec_material_id, score))
        
        return recommendations
    
    
    def _create_user_profile(self, user_history, df):
        """
        Membuat profil user berdasarkan riwayat materi yang diakses
    
        Parameters:
        - user_history: List material_id yang pernah diakses user
        - df: DataFrame lengkap untuk mendapatkan fitur materi
    
        Returns:
        - user_profile: Vektor TF-IDF yang merepresentasikan preferensi user
        """
        # Dapatkan index materi yang pernah diakses user
        history_indices = []
        for material_id in user_history:
            try:
                idx = self.indices[material_id]  # Dapatkan indeks berdasarkan material_id
                history_indices.append(idx)
            except KeyError:
                continue

        # Pastikan history_indices tidak kosong dan memiliki bentuk yang benar
        if not history_indices:
            return None  # Jika tidak ada materi yang bisa diakses, return None

        # Filter untuk memastikan semua indeks adalah integer dan tidak memiliki nilai yang tidak diinginkan
        history_indices = [idx for idx in history_indices if isinstance(idx, int)]

        # Pastikan history_indices adalah array numpy yang valid
        if len(history_indices) > 0:
            history_indices = np.array(history_indices)
        
            # Hitung mean hanya jika ada history
            user_profile = self.tfidf_matrix[history_indices].mean(axis=0)
            return user_profile.A1  # Convert to dense array
        return None
    
    def save_model(self, save_path='models/recommenders/content_based'):
        """
        Menyimpan model yang sudah dilatih
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'cosine_sim': self.cosine_sim,
            'indices': self.indices,
            'material_ids': self.material_ids,
            'n_recommendations': self.n_recommendations
        }
        
        joblib.dump(model_data, f'{save_path}/content_model.joblib')
        print("Model Content-Based Filtering berhasil disimpan!")
    
    @classmethod
    def load_model(cls, load_path='models/recommenders/content_based/content_model.joblib'):
        """
        Memuat model yang sudah disimpan
        """
        model_data = joblib.load(load_path)
        
        model = cls(n_recommendations=model_data['n_recommendations'])
        model.tfidf_vectorizer = model_data['tfidf_vectorizer']
        model.tfidf_matrix = model_data['tfidf_matrix']
        model.cosine_sim = model_data['cosine_sim']
        model.indices = model_data['indices']
        model.material_ids = model_data['material_ids']
        
        return model
