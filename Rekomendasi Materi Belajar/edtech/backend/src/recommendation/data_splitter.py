# backend/src/recommendation/data_splitter.py
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

class DataSplitter:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
    
    def split_data(self, df):
        """
        Membagi data menjadi train dan test set untuk rekomendasi
        
        Parameters:
        - df: DataFrame yang sudah diproses
        
        Returns:
        - train_data: Data untuk training
        - test_data: Data untuk testing
        - user_item_matrix: Matriks interaksi user-item
        """
        # Cek kolom yang ada di data
        print("Kolom-kolom dalam data:", df.columns)  # Menambahkan pengecekan kolom
        
        # Pastikan data sudah diacak
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # Membagi data secara stratifikasi berdasarkan student_id
        train_data, test_data = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df['student_id']
        )
        
        # Membuat user-item matrix untuk collaborative filtering
        user_item_matrix = self._create_user_item_matrix(df)
        
        return train_data, test_data, user_item_matrix
    
    def _create_user_item_matrix(self, df):
        # 1. Hitung composite engagement score dengan handling missing values
        df['engagement_score'] = (
            0.4 * df['engagement_score'].fillna(0).clip(lower=0) +
            0.3 * df['completion_rate'].fillna(0).clip(0, 1) +
            0.2 * df['material_rating'].fillna(3).clip(1, 5) / 5 +  # normalisasi ke 0-1
            0.1 * df['quiz_score'].fillna(50).clip(0, 100) / 100  # normalisasi ke 0-1
        )
        
        # 2. Normalisasi yang lebih aman untuk dataset kecil
        def safe_normalize(x):
            x_min = x.min()
            x_max = x.max()
            if x_max == x_min:
                return x * 0 + 0.5  # beri nilai netral jika semua sama
            return (x - x_min) / (x_max - x_min)
        
        df['engagement_score'] = df.groupby('student_id')['engagement_score'].transform(safe_normalize)
        
        # 3. Buat matriks user-item dengan kriteria yang lebih longgar
        user_item_matrix = df.pivot_table(
            index='student_id',
            columns='material_type_encoded',
            values='engagement_score',
            aggfunc='mean',
            fill_value=0
        )
        
        # 4. Filter yang sangat longgar untuk dataset kecil
        min_user_interactions = 1  # Minimal 1 interaksi per user
        min_item_interactions = 1  # Minimal 1 interaksi per item
        
        # Hitung interaksi
        user_interactions = (user_item_matrix > 0).sum(axis=1)
        item_interactions = (user_item_matrix > 0).sum(axis=0)
        
        # Filter dengan logging
        print(f"Sebelum filter - Users: {len(user_interactions)}, Items: {len(item_interactions)}")
        print(f"Kriteria filter - Min user interaksi: {min_user_interactions}, Min item interaksi: {min_item_interactions}")
        
        # Terapkan filter yang sangat longgar
        filtered_users = user_interactions[user_interactions >= min_user_interactions].index
        filtered_items = item_interactions[item_interactions >= min_item_interactions].index
        
        user_item_matrix = user_item_matrix.loc[filtered_users, filtered_items]
        
        # 5. Tambahkan pseudo-interaksi jika matriks terlalu sparse
        if user_item_matrix.shape[0] < 10 or user_item_matrix.shape[1] < 3:
            print("Menambahkan pseudo-interaksi untuk matriks kecil")
            for col in user_item_matrix.columns:
                if user_item_matrix[col].sum() == 0:
                    user_item_matrix[col].iloc[0] = 0.1  # Tambahkan interaksi kecil
        
        # Logging akhir
        print(f"Sesudah filter - Users: {user_item_matrix.shape[0]}, Items: {user_item_matrix.shape[1]}")
        density = (user_item_matrix > 0).mean().mean()
        print(f"Kepadatan matriks: {density:.2%}")
        
        return user_item_matrix

    def save_split(self, train_data, test_data, save_dir='data/recommendations'):
        """
        Menyimpan data yang sudah dibagi
        
        Parameters:
        - train_data: Data training
        - test_data: Data testing
        - save_dir: Direktori penyimpanan
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        train_data.to_csv(f'{save_dir}/train_data.csv', index=False)
        test_data.to_csv(f'{save_dir}/test_data.csv', index=False)
        
        print("Data berhasil dibagi dan disimpan!")
