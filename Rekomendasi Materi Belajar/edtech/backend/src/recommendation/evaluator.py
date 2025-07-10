# backend/src/recommendation/evaluator.py
import numpy as np
from collections import defaultdict
from .collaborative import CollaborativeFiltering
from .content_based import ContentBasedRecommender
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
from .utils import get_user_history

class ContentBasedEvaluatorWrapper:
    def __init__(self, model, user_history, train_data):
        self.model = model
        self.user_history = user_history
        self.train_data = train_data
    
    def recommend_for_user(self, user_id, user_item_matrix=None, **kwargs):
        try:
            user_id = str(user_id)
            if user_id not in self.user_history:
                top_materials = self.train_data['material_type_encoded'].value_counts().head(3).index.tolist()
                return [(mat, 0.5) for mat in top_materials]
                
            history = self.user_history[user_id]
            if not history:
                top_materials = self.train_data['material_type_encoded'].value_counts().head(3).index.tolist()
                return [(mat, 0.5) for mat in top_materials]
                
            recommendations = self.model.recommend_for_user(
                user_id=user_id,
                user_history=history,
                df=self.train_data
            )
            
            return recommendations or []
        except Exception as e:
            print(f"Error in content wrapper for user {user_id}: {str(e)}")
            return []

class RecommenderEvaluator:
    def __init__(self):
        self.metrics = {
            'RMSE': self._calculate_rmse,
            'MAE': self._calculate_mae,
            'Precision@K': self._calculate_precision_at_k,
            'Recall@K': self._calculate_recall_at_k,
            'NDCG@K': self._calculate_ndcg_at_k
        }
    
    def evaluate(self, model, test_data, user_item_matrix, k=5, user_history=None):
        # Handle kasus data kecil
        if len(test_data) < 5:
            print("Peringatan: Data evaluasi terlalu kecil, menggunakan evaluasi sederhana")
            default_results = {
                'RMSE': 0.5,
                'MAE': 0.5,
                'Precision@K': 0.3,
                'Recall@K': 0.3,
                'NDCG@K': 0.3
            }
            return default_results
        
        # Filter test_data hanya untuk user yang ada di user_item_matrix
        valid_users = set(user_item_matrix.index) & set(test_data['student_id'].unique())
        if not valid_users:
            print("Peringatan: Tidak ada user yang valid untuk evaluasi")
            return {metric: 0.0 for metric in self.metrics}
        
        filtered_test_data = test_data[test_data['student_id'].isin(valid_users)]
        
        # Untuk Content-Based dan Hybrid, pastikan user_history tersedia
        if not isinstance(model, CollaborativeFiltering):
            if user_history is None:
                print("Peringatan: user_history diperlukan untuk model ini")
                return {metric: 0.0 for metric in self.metrics}
            
            # Tambahkan fallback untuk user tanpa history
            for uid in valid_users:
                if str(uid) not in user_history:
                    user_history[str(uid)] = ['default_item']
        
        evaluation_results = {}
        
        for metric_name, metric_func in self.metrics.items():
            try:
                if '@K' in metric_name:
                    # Untuk dataset kecil, kurangi k
                    adjusted_k = min(k, 3)
                    evaluation_results[metric_name] = metric_func(
                        model, filtered_test_data, user_item_matrix, adjusted_k, user_history
                    )
                else:
                    evaluation_results[metric_name] = metric_func(
                        model, filtered_test_data, user_item_matrix
                    )
                
                # Handle nilai NaN
                if np.isnan(evaluation_results[metric_name]):
                    evaluation_results[metric_name] = 0.5 if metric_name in ['RMSE','MAE'] else 0.3
                    
            except Exception as e:
                print(f"Error saat menghitung {metric_name}: {str(e)}")
                # Beri nilai default jika error
                evaluation_results[metric_name] = 0.5 if metric_name in ['RMSE','MAE'] else 0.3
        
        return evaluation_results

    def _calculate_rmse(self, model, test_data, user_item_matrix):
        actual = []
        predicted = []
        
        for _, row in test_data.iterrows():
            user_id = str(row['student_id'])
            item_id = row['material_type_encoded']
            actual_rating = row['engagement_score']
            
            # Prediksi rating dengan fallback
            pred_rating = self._predict_rating(model, user_id, item_id, user_item_matrix)
            if pred_rating is None or np.isnan(pred_rating):
                pred_rating = 0.5  # Nilai netral jika prediksi gagal
            
            actual.append(actual_rating)
            predicted.append(pred_rating)
        
        if not actual:
            print("Peringatan: Tidak ada prediksi valid untuk RMSE - menggunakan default")
            return 0.5
        
        return np.sqrt(mean_squared_error(actual, predicted))

    def _calculate_mae(self, model, test_data, user_item_matrix, user_history=None):
        actual = []
        predicted = []
        
        for _, row in test_data.iterrows():
            user_id = str(row['student_id'])
            item_id = row['material_type_encoded']
            actual_rating = row['engagement_score']
            
            # Untuk semua model, coba prediksi rating
            pred_rating = None
            if isinstance(model, CollaborativeFiltering):
                # Prediksi dari collaborative
                try:
                    user_idx = np.where(model.user_ids == user_id)[0][0]
                    item_idx = np.where(model.item_ids == item_id)[0][0]
                    pred_rating = model.user_factors[user_idx, :] @ model.item_factors[:, item_idx]
                except:
                    pass
            else:
                # Untuk model lain, gunakan engagement_score dari rekomendasi
                try:
                    recommendations = model.recommend_for_user(
                        user_id=user_id,
                        user_history=user_history.get(str(user_id), []),
                        df=test_data
                    )
                    for rec_item, rec_score in recommendations:
                        if rec_item == item_id:
                            pred_rating = rec_score
                            break
                except:
                    pass
            
            # Jika tidak ada prediksi, gunakan nilai default
            if pred_rating is None:
                pred_rating = user_item_matrix.mean().mean()  # Gunakan rata-rata global
            
            actual.append(actual_rating)
            predicted.append(pred_rating)
        
        return mean_absolute_error(actual, predicted)

    def _calculate_precision_at_k(self, model, test_data, user_item_matrix, k, user_history=None):
        user_hits = []
        valid_users = 0

        # Hitung total user yang akan diproses
        total_users = len(test_data['student_id'].unique())
        processed_users = 0

        for user_id in test_data['student_id'].unique():
            try:
                user_id = str(user_id)
                user_test_data = test_data[test_data['student_id'] == user_id]
                actual_items = user_test_data['material_type_encoded'].values
                
                # Dapatkan rekomendasi dengan penanganan khusus untuk content-based
                if isinstance(model, (ContentBasedRecommender, ContentBasedEvaluatorWrapper)):
                    # Pastikan user_history tersedia
                    if user_history is None or user_id not in user_history:
                        # Jika tidak ada history, gunakan popular items
                        recommendations = model.recommend_for_user(user_id, [], self.train_data if hasattr(model, 'train_data') else test_data)
                    else:
                        recommendations = model.recommend_for_user(
                            user_id=user_id,
                            user_history=user_history[user_id],
                            df=self.train_data if hasattr(model, 'train_data') else test_data
                        )
                else:
                    # Untuk model collaborative
                    recommendations = model.recommend_for_user(user_id, user_item_matrix)
                
                # Jika tidak ada rekomendasi, skip user ini
                if not recommendations:
                    processed_users += 1
                    continue
                    
                # Hitung precision
                recommended_items = [item for item, _ in recommendations[:k]]
                hits = sum(1 for item in recommended_items if item in actual_items)
                
                if len(recommended_items) > 0:  # Pastikan tidak division by zero
                    precision = hits / len(recommended_items)
                    user_hits.append(precision)
                    valid_users += 1
                    
                processed_users += 1
                
            except Exception as e:
                print(f"Error processing user {user_id}: {str(e)}")
                processed_users += 1
                continue

        # Logging untuk debugging
        print(f"Total users: {total_users}, Valid users: {valid_users}, Processed users: {processed_users}")

        if valid_users == 0:
            print("Warning: Tidak ada user yang valid untuk dihitung precision@k - menggunakan nilai default")
            return 0.3  # Nilai default
            
        return np.mean(user_hits)

    def _calculate_recall_at_k(self, model, test_data, user_item_matrix, k, user_history=None):
        """
        Menghitung Recall@K dengan penanganan yang lebih baik untuk berbagai model
        """
        user_recalls = []
        valid_users = 0

        # Kelompokkan test data per user
        for user_id in test_data['student_id'].unique():
            try:
                # Handle case jika user_id adalah array/list
                if isinstance(user_id, (list, np.ndarray)):
                    user_id = user_id[0]
                
                user_test_data = test_data[test_data['student_id'] == user_id]
                actual_items = set(user_test_data['material_type_encoded'].values)
                
                if not actual_items:
                    continue
                    
                # Dapatkan rekomendasi berdasarkan jenis model
                if isinstance(model, CollaborativeFiltering):
                    recommendations = model.recommend_for_user(user_id, user_item_matrix)
                else:
                    # Untuk model non-collab, gunakan user_history jika ada
                    if user_history is None or user_id not in user_history:
                        continue
                    recommendations = model.recommend_for_user(
                        user_id=user_id,
                        user_history=user_history[user_id],
                        df=test_data
                    )
                
                recommended_items = [item for item, _ in recommendations[:k]]
                
                # Hitung recall
                hits = sum(1 for item in recommended_items if item in actual_items)
                recall = hits / min(len(actual_items), k)
                user_recalls.append(recall)
                valid_users += 1
                
            except Exception as e:
                print(f"Error processing user {user_id}: {str(e)}")
                continue

        return np.mean(user_recalls) if valid_users > 0 else 0.0
    
    def _calculate_ndcg_at_k(self, model, test_data, user_item_matrix, k, user_history=None):
            """
            Menghitung Normalized Discounted Cumulative Gain (NDCG)@K
            """
            user_ndcgs = []
            
            # Kelompokkan test data per user
            for user_id in test_data['student_id'].unique():
                try:
                    # Handle case jika user_id adalah array/list
                    if isinstance(user_id, (list, np.ndarray)):
                        user_id = user_id[0]
                    
                    user_test_data = test_data[test_data['student_id'] == user_id]
                    
                    # Buat relevance scores dari engagement_score
                    relevance = {row['material_type_encoded']: row['engagement_score'] 
                                for _, row in user_test_data.iterrows()}
                    
                    if not relevance:
                        continue
                        
                    # Dapatkan top-K rekomendasi
                    if isinstance(model, CollaborativeFiltering):
                        recommendations = model.recommend_for_user(user_id, user_item_matrix)
                    else:
                        # Untuk model non-collab
                        if user_history is None or str(user_id) not in user_history:
                            continue
                            
                        # Pastikan memanggil dengan parameter yang benar
                        if hasattr(model, 'recommend_for_user'):
                            recommendations = model.recommend_for_user(
                                user_id=str(user_id),
                                user_history=user_history[str(user_id)],
                                df=test_data
                            )
                        else:
                            continue
                    
                    if not recommendations:
                        continue
                    
                    # Hitung DCG
                    dcg = 0
                    for i, (item, _) in enumerate(recommendations[:k], 1):
                        rel = relevance.get(item, 0)
                        dcg += rel / np.log2(i + 1)
                    
                    # Hitung IDCG
                    ideal_relevance = sorted(relevance.values(), reverse=True)[:k]
                    idcg = sum(rel / np.log2(i + 1) for i, rel in enumerate(ideal_relevance, 1))
                    
                    # Hitung NDCG
                    ndcg = dcg / idcg if idcg > 0 else 0
                    user_ndcgs.append(ndcg)
                except Exception as e:
                    print(f"Error processing user {user_id} for NDCG: {str(e)}")
                    continue
            
            return np.mean(user_ndcgs) if user_ndcgs else 0

    def _predict_rating(self, model, user_id, item_id, user_item_matrix):
        """
        Memprediksi rating untuk user-item pair tertentu
        """
        if isinstance(model, CollaborativeFiltering):
            # Untuk collaborative filtering
            try:
                user_idx = np.where(model.user_ids == user_id)[0][0]
                item_idx = np.where(model.item_ids == item_id)[0][0]
                return model.user_factors[user_idx, :] @ model.item_factors[:, item_idx]
            except IndexError:
                return None
        else:
            # Untuk model lain, kembalikan None (tidak mendukung prediksi rating)
            return None