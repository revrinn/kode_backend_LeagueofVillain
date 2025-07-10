# backend/src/performance_prediction/predictor.py

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Union, Dict, List, Optional
import xgboost as xgb
import shap
from datetime import datetime

class PerformancePredictor:
    def __init__(self, model_path: str, preprocessor_path: Optional[str] = None):
        """
        Inisialisasi predictor dengan model dan preprocessor
        
        Parameters:
            model_path: Path ke model yang sudah dilatih
            preprocessor_path: Path ke preprocessor (opsional)
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.shap_explainer = None
        self.logger = self._setup_logger()
        self._load_components()
        
    def _setup_logger(self):
        """Setup logger untuk predictor"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _load_components(self):
        """Memuat model dan preprocessor"""
        try:
            # Load model
            self.model = joblib.load(self.model_path)
            self.logger.info(f"Model berhasil dimuat dari {self.model_path}")
            
            # Load preprocessor jika ada
            if self.preprocessor_path:
                processor_data = joblib.load(self.preprocessor_path)
                self.preprocessor = processor_data['preprocessor']
                self.feature_names = processor_data['feature_names']
                self.logger.info(f"Preprocessor berhasil dimuat dari {self.preprocessor_path}")
            
            # Setup SHAP explainer
            self._setup_shap_explainer()
            
        except Exception as e:
            self.logger.error(f"Gagal memuat komponen: {str(e)}")
            raise
        
    def _setup_shap_explainer(self):
        """Mempersiapkan SHAP explainer untuk interpretasi"""
        try:
            if hasattr(self.model, 'predict_proba'):
                self.shap_explainer = shap.Explainer(self.model)
            else:
                self.shap_explainer = shap.Explainer(self.model)
            self.logger.info("SHAP explainer berhasil diinisialisasi")
        except Exception as e:
            self.logger.warning(f"Tidak dapat menginisialisasi SHAP explainer: {str(e)}")
            self.shap_explainer = None
    
    def _prepare_input(self, input_data: Union[Dict, List[Dict]], return_dataframe: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """
        Mempersiapkan input data untuk prediksi

        Parameters:
            input_data: Input data dalam bentuk dict atau list of dicts
            return_dataframe: Jika True kembalikan DataFrame, jika False kembalikan array

        Returns:
            Data yang sudah diproses dalam bentuk array atau DataFrame
        """
        # Konversi input ke DataFrame
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            input_df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data.copy()
        else:
            raise ValueError("Input harus berupa dict, list of dicts, atau DataFrame")

        # Validasi kolom
        if self.feature_names is not None:
            missing_cols = set(self.feature_names) - set(input_df.columns)
            if missing_cols:
                raise ValueError(f"Kolom berikut tidak ditemukan dalam input: {missing_cols}")

            # Urutkan kolom sesuai dengan yang diharapkan model
            input_df = input_df[self.feature_names]

        # Preprocess data jika ada preprocessor
        if self.preprocessor is not None:
            processed_data = self.preprocessor.transform(input_df)
        else:
            processed_data = input_df.values if not return_dataframe else input_df

        return processed_data if not return_dataframe else input_df

    
    def predict(self, input_data: Union[Dict, List[Dict]], 
                return_contributions: bool = False) -> Dict:
        """
        Membuat prediksi dari input data dengan opsi interpretasi
        
        Parameters:
            input_data: Input data dalam bentuk dict atau list of dicts
            return_contributions: Jika True, kembalikan kontribusi fitur
            
        Returns:
            Dict berisi prediksi dan informasi tambahan
        """
        start_time = datetime.now()
        
        try:
            # Persiapkan input
            processed_input = self._prepare_input(input_data)
            
            # Buat prediksi
            if isinstance(self.model, xgb.Booster):
                dmatrix = xgb.DMatrix(processed_input)
                predictions = self.model.predict(dmatrix)
            else:
                predictions = self.model.predict(processed_input)
            
            # Hitung confidence interval (simplified)
            if hasattr(self.model, 'predict_quantiles'):
                quantiles = self.model.predict_quantiles(processed_input, quantiles=(0.025, 0.975))
                confidence_intervals = list(zip(quantiles[0], quantiles[1]))
            else:
                # Fallback untuk model tanpa quantile prediction
                std_dev = np.std(predictions)
                confidence_intervals = [(p - 1.96*std_dev, p + 1.96*std_dev) for p in predictions]
            
            # Hitung feature contributions jika diminta
            feature_contributions = None
            if return_contributions and self.shap_explainer is not None:
                feature_contributions = self._calculate_feature_contributions(processed_input)
            
            # Hitung waktu eksekusi
            exec_time = (datetime.now() - start_time).total_seconds()
            
            # Format hasil
            if isinstance(predictions, np.ndarray) and predictions.ndim == 1:
                predictions = predictions.tolist()
            
            result = {
                'predictions': predictions,
                'confidence_intervals': confidence_intervals,
                'execution_time_seconds': exec_time,
                'timestamp': start_time.isoformat()
            }
            
            if feature_contributions is not None:
                result['feature_contributions'] = feature_contributions
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error dalam prediksi: {str(e)}")
            raise
    
    def _calculate_feature_contributions(self, processed_input: np.ndarray) -> List[Dict]:
        """
        Menghitung kontribusi fitur menggunakan SHAP values
        
        Parameters:
            processed_input: Input data yang sudah diproses
            
        Returns:
            List berisi kontribusi setiap fitur untuk setiap sampel
        """
        if self.shap_explainer is None:
            return None
            
        # Hitung SHAP values
        shap_values = self.shap_explainer(processed_input)
        
        # Format hasil
        contributions = []
        for i in range(len(processed_input)):
            sample_contributions = []
            
            for j, feature_name in enumerate(self.feature_names):
                sample_contributions.append({
                    'feature': feature_name,
                    'value': processed_input[i][j] if isinstance(processed_input, np.ndarray) else processed_input.iloc[i][j],
                    'contribution': float(shap_values.values[i][j]),
                    'abs_contribution': float(np.abs(shap_values.values[i][j]))
                })
            
            # Urutkan berdasarkan kontribusi absolut terbesar
            sample_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
            contributions.append(sample_contributions)
        
        return contributions
    
    def batch_predict(self, input_data: List[Dict], batch_size: int = 100,
                     return_contributions: bool = False) -> Dict:
        """
        Membuat prediksi dalam batch untuk efisiensi
        
        Parameters:
            input_data: List of dicts berisi input data
            batch_size: Ukuran batch untuk prediksi
            return_contributions: Jika True, kembalikan kontribusi fitur
            
        Returns:
            Dict berisi hasil prediksi untuk semua sampel
        """
        start_time = datetime.now()
        total_samples = len(input_data)
        results = []
        
        self.logger.info(f"Memulai batch prediction untuk {total_samples} sampel (batch_size={batch_size})")
        
        for i in range(0, total_samples, batch_size):
            batch = input_data[i:i+batch_size]
            try:
                batch_result = self.predict(batch, return_contributions)
                results.extend(batch_result['predictions'])
            except Exception as e:
                self.logger.error(f"Error pada batch {i//batch_size}: {str(e)}")
                raise
        
        exec_time = (datetime.now() - start_time).total_seconds()
        avg_time_per_sample = exec_time / total_samples
        
        self.logger.info(
            f"Batch prediction selesai. Total waktu: {exec_time:.2f} detik "
            f"({avg_time_per_sample:.4f} detik/sampel)"
        )
        
        return {
            'predictions': results,
            'total_samples': total_samples,
            'total_time_seconds': exec_time,
            'avg_time_per_sample': avg_time_per_sample,
            'timestamp': start_time.isoformat()
        }
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluasi model pada dataset test
        
        Parameters:
            X_test: Data fitur test
            y_test: Target test
            
        Returns:
            Dict berisi metrik evaluasi
        """
        from .evaluator import PerformanceEvaluator
        
        evaluator = PerformanceEvaluator(y_test, self.predict(X_test)['predictions'], 
                                       self.model, X_test)
        return evaluator.metrics
    
    def save_predictor(self, save_dir: str):
        """
        Menyimpan objek predictor untuk penggunaan nanti
        
        Parameters:
            save_dir: Direktori untuk menyimpan predictor
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Nama file berdasarkan timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = save_path / f"predictor_{timestamp}.pkl"
        
        # Simpan objek predictor
        joblib.dump(self, save_file)
        self.logger.info(f"Predictor disimpan di: {save_file}")
        
        return str(save_file)