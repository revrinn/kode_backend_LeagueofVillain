# backend/src/performance_prediction/data_processor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os
from datetime import datetime
from pathlib import Path
import logging
import json

class PerformanceDataProcessor:
    def __init__(self, data_path, config_path=None):
        self.data_path = data_path
        self.config_path = config_path
        self.features = None
        self.target = None
        self.preprocessor = None
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def load_data(self):
        """Memuat data dengan penanganan error yang lebih baik"""
        try:
            data = pd.read_csv(self.data_path)
            
            # Log informasi dasar data
            self.logger.info(f"Data berhasil dimuat. Shape: {data.shape}")
            self.logger.info(f"Kolom yang tersedia: {list(data.columns)}")
            self.logger.info(f"Contoh data:\n{data.head(2)}")
            
            return data
        except Exception as e:
            self.logger.error(f"Gagal memuat data: {str(e)}")
            raise
            
    def prepare_features_target(self, data, target_col='quiz_score'):
        """
        Menyiapkan fitur dan target dengan penanganan data yang lebih komprehensif
        """
        try:
            # Load feature configuration if available
            if self.config_path:
                with open(self.config_path) as f:
                    config = json.load(f)
                relevant_features = config.get('features', [])
            else:
                # Default features
                relevant_features = [
                    'grade', 'tech_savvy', 'duration_minutes', 'engagement_score',
                    'completion_rate', 'material_rating', 'interaction_duration',
                    'material_engagement_score', 'feature_engagement', 'jam_belajar',
                    'hari_dalam_minggu', 'akhir_pekan', 'efisiensi_belajar',
                    'rasio_penyelesaian', 'interaksi_total', 'preferensi_materi',
                    'jumlah_pengakses', 'engagement_rata2', 'performance_label_encoded',
                    'learning_speed_encoded', 'student_feedback_encoded',
                    'achievement_status_encoded'
                ]
            
            # Tambahkan fitur interaksi baru
            data['efisiensi_engagement'] = data['engagement_score'] / (data['duration_minutes'] + 1e-6)
            data['learning_consistency'] = data['completion_rate'] * data['material_rating']
            relevant_features.extend(['efisiensi_engagement', 'learning_consistency'])
            
            # Pastikan kolom target ada
            if target_col not in data.columns:
                raise ValueError(f"Kolom target '{target_col}' tidak ditemukan")
                
            # Handle missing values
            data[relevant_features] = data[relevant_features].fillna(data[relevant_features].median())
            
            self.features = data[relevant_features]
            self.target = data[target_col]
            
            # Setup preprocessing pipeline
            numeric_features = self.features.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = self.features.select_dtypes(include=['object', 'category']).columns
            
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())  # Lebih robust terhadap outlier
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            
            return self.features, self.target
            
        except Exception as e:
            self.logger.error(f"Error dalam menyiapkan fitur: {str(e)}")
            raise
    
    def split_data(self, test_size=0.2, val_size=0.2, random_state=42):
        """Membagi data menjadi train, validation, dan test set"""
        try:
            if self.features is None or self.target is None:
                raise ValueError("Fitur atau target belum disiapkan")
                
            # Bagi data menjadi train+val dan test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                self.features, self.target, 
                test_size=test_size, 
                random_state=random_state
            )
            
            # Bagi train_val menjadi train dan validation
            val_size_adjusted = val_size / (1 - test_size)  # Adjust untuk ukuran asli dataset
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=val_size_adjusted,
                random_state=random_state
            )
            
            # Preprocess data
            X_train = self.preprocessor.fit_transform(X_train)
            X_val = self.preprocessor.transform(X_val)
            X_test = self.preprocessor.transform(X_test)
            
            # Validasi data
            self._validate_data(X_train, y_train)
            self._validate_data(X_val, y_val)
            self._validate_data(X_test, y_test)
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            self.logger.error(f"Error dalam membagi data: {str(e)}")
            raise

    def _validate_data(self, X, y):
        """Validasi kualitas data"""
        if isinstance(X, np.ndarray):
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                raise ValueError("Data mengandung NaN atau infinity")
        if len(X) != len(y):
            raise ValueError("Jumlah sampel X dan y tidak sama")
        if len(y) == 0:
            raise ValueError("Data target kosong")
            
    def save_processor(self, save_dir):
        """Menyimpan processor dan preprocessing pipeline"""
        try:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = Path(save_dir) / f"data_processor_{timestamp}.pkl"
            
            # Simpan seluruh objek processor
            joblib.dump({
                'processor': self,
                'preprocessor': self.preprocessor,
                'feature_names': list(self.features.columns) if self.features is not None else None
            }, save_path)
            
            self.logger.info(f"Processor disimpan di: {save_path}")
            return str(save_path)
            
        except Exception as e:
            self.logger.error(f"Gagal menyimpan processor: {str(e)}")
            raise