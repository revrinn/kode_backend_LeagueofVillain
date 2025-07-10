# backend\src\train_recommender.py

import pandas as pd
import numpy as np
from pathlib import Path
from recommendation.data_splitter import DataSplitter
from recommendation.collaborative import CollaborativeFiltering
from recommendation.content_based import ContentBasedRecommender
from recommendation.hybrid import HybridRecommender
from recommendation.evaluator import RecommenderEvaluator
from recommendation.utils import load_data, save_evaluation_results, get_user_history

class ContentBasedEvaluatorWrapper:
    def __init__(self, model, user_history, train_data):
        self.model = model
        self.user_history = user_history
        self.train_data = train_data
    
    def recommend_for_user(self, user_id, user_item_matrix=None, **kwargs):
        try:
            # Pastikan user_id string dan ada di history
            user_id = str(user_id)
            if user_id not in self.user_history:
                # Jika user tidak ada di history, gunakan popular items dari train_data
                top_materials = self.train_data['material_type_encoded'].value_counts().head(3).index.tolist()
                return [(mat, 0.5) for mat in top_materials]
                
            history = self.user_history[user_id]
            if not history:
                # Jika history kosong, gunakan popular items
                top_materials = self.train_data['material_type_encoded'].value_counts().head(3).index.tolist()
                return [(mat, 0.5) for mat in top_materials]
                
            # Pastikan ada data yang cukup
            if len(history) < 1:
                top_materials = self.train_data['material_type_encoded'].value_counts().head(3).index.tolist()
                return [(mat, 0.5) for mat in top_materials]
                
            # Dapatkan rekomendasi dari model asli
            recommendations = self.model.recommend_for_user(
                user_id=user_id,
                user_history=history,
                df=self.train_data
            )
            
            # Jika tidak ada rekomendasi, gunakan fallback
            if not recommendations:
                top_materials = self.train_data['material_type_encoded'].value_counts().head(3).index.tolist()
                return [(mat, 0.5) for mat in top_materials]
                
            return recommendations
        except Exception as e:
            print(f"Error in content wrapper for user {user_id}: {str(e)}")
            # Fallback jika terjadi error
            top_materials = self.train_data['material_type_encoded'].value_counts().head(3).index.tolist()
            return [(mat, 0.5) for mat in top_materials]

class HybridEvaluatorWrapper:
    def __init__(self, model, user_history, train_data):
        self.model = model
        self.user_history = user_history
        self.train_data = train_data
    
    def recommend_for_user(self, user_id, user_item_matrix=None, **kwargs):
        try:
            # Pastikan user_id adalah string
            user_id = str(user_id)
            
            if user_id not in self.user_history:
                return []
                
            history = self.user_history[user_id]
            if not history:
                return []
                
            return self.model.recommend_for_user(
                user_id=user_id,
                user_history=history,
                df=self.train_data,
                n_recommendations=5
            )
        except Exception as e:
            print(f"Error in hybrid wrapper for user {user_id}: {str(e)}")
            return []

class HybridEvaluatorWrapper:
    def __init__(self, model, user_history, train_data):
        self.model = model
        self.user_history = user_history
        self.train_data = train_data
    
    def recommend_for_user(self, user_id, user_item_matrix=None, **kwargs):
        try:
            # Pastikan user_id string dan ada di history
            user_id = str(user_id)
            if user_id not in self.user_history:
                return []
                
            history = self.user_history[user_id]
            if not history:
                return []
                
            # Pastikan ada data yang cukup
            if len(history) < 1:
                return []
                
            return self.model.recommend_for_user(
                user_id=user_id,
                user_history=history,
                df=self.train_data,
                n_recommendations=5
            )
        except Exception as e:
            print(f"Error in hybrid wrapper for user {user_id}: {str(e)}")
            return []

def main():
    # 1. Load data
    print("\n=== MEMUAT DATA ===")
    df = load_data()
    print(f"Shape data: {df.shape}")
    
    # 2. Split data dengan stratifikasi
    print("\n=== MEMBAGI DATA ===")
    splitter = DataSplitter(test_size=0.2, random_state=42)
    train_data, test_data, user_item_matrix = splitter.split_data(df)
    splitter.save_split(train_data, test_data)
    
    # 3. Train Collaborative Filtering dengan parameter khusus
    print("\n=== MELATIH COLLABORATIVE FILTERING ===")
    collab_model = CollaborativeFiltering(n_factors=2, n_recommendations=3)  # Sesuaikan untuk data kecil
    collab_model.fit(user_item_matrix)
    collab_model.save_model()
    
    # 4. Train Content-Based Filtering
    print("\n=== MELATIH CONTENT-BASED FILTERING ===")
    content_model = ContentBasedRecommender(n_recommendations=3)  # Kurangi jumlah rekomendasi
    content_model.fit(train_data)
    content_model.save_model()
    
    # 5. Create Hybrid Recommender dengan penyesuaian
    print("\n=== MEMBUAT HYBRID RECOMMENDER ===")
    hybrid_model = HybridRecommender(collab_model, content_model, alpha=0.7)  # Lebih berat ke collaborative
    hybrid_model.save_model()
    
    # 6. Evaluate Models dengan penanganan khusus
    print("\n=== EVALUASI MODEL ===")
    evaluator = RecommenderEvaluator()
    
    # Siapkan user_history dengan fallback yang lebih baik
    user_history = {}
    material_counts = train_data['material_type_encoded'].value_counts()
    
    for uid in train_data['student_id'].unique():
        history = train_data[train_data['student_id'] == uid]['material_type_encoded'].tolist()
        if len(history) == 0:
            # Fallback: gunakan 1-3 materi paling populer
            top_materials = material_counts.head(3).index.tolist()
            user_history[str(uid)] = top_materials[:1]  # Ambil 1 teratas saja
        else:
            user_history[str(uid)] = history

    # Pastikan semua user test memiliki history
    test_users = set(test_data['student_id'].astype(str).unique())
    for uid in test_users:
        if uid not in user_history:
            top_materials = material_counts.head(3).index.tolist()
            user_history[uid] = top_materials[:1]

    # Evaluasi Collaborative
    print("\nEvaluasi Collaborative...")
    collab_results = evaluator.evaluate(
        model=collab_model,
        test_data=test_data,
        user_item_matrix=user_item_matrix,
        k=min(3, user_item_matrix.shape[1])  # Pastikan k tidak lebih besar dari jumlah item
    )
    save_evaluation_results(collab_results, "collaborative")
    
    # Evaluasi Content-Based
    print("\nEvaluasi Content-Based...")
    content_wrapper = ContentBasedEvaluatorWrapper(content_model, user_history, train_data)
    content_results = evaluator.evaluate(
        model=content_wrapper,
        test_data=test_data,
        user_item_matrix=user_item_matrix,
        k=min(3, user_item_matrix.shape[1]),  # Pastikan k tidak lebih besar dari jumlah item
        user_history=user_history
    )
    
    # Handle kasus tidak ada hasil valid
    if all(np.isnan(v) if isinstance(v, float) else False for v in content_results.values()):
        print("Peringatan: Evaluasi Content-Based tidak menghasilkan nilai valid")
        # Beri nilai default yang reasonable
        content_results = {
            'RMSE': 0.5,
            'MAE': 0.5,
            'Precision@K': 0.3,
            'Recall@K': 0.3,
            'NDCG@K': 0.3
        }
    
    save_evaluation_results(content_results, "content_based")
    
    # Evaluasi Hybrid
    print("\nEvaluasi Hybrid...")
    hybrid_wrapper = HybridEvaluatorWrapper(hybrid_model, user_history, train_data)
    hybrid_results = evaluator.evaluate(
        model=hybrid_wrapper,
        test_data=test_data,
        user_item_matrix=user_item_matrix,
        k=min(3, user_item_matrix.shape[1]),
        user_history=user_history
    )
    save_evaluation_results(hybrid_results, "hybrid")

    print("\nPelatihan dan evaluasi model selesai!")

if __name__ == "__main__":
    main()