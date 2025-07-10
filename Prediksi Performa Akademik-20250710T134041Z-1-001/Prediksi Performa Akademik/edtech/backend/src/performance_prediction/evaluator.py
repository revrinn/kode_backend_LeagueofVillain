# backend/src/performance_prediction/evaluator.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    mean_absolute_percentage_error
)
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path
import json
import shap

class PerformanceEvaluator:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, model=None, X_test=None):
        """
        Inisialisasi evaluator dengan tambahan SHAP values dan model interpretability
        
        Parameters:
            y_true (np.ndarray): Nilai sebenarnya
            y_pred (np.ndarray): Nilai prediksi
            model (optional): Model yang sudah dilatih untuk interpretasi
            X_test (optional): Data fitur untuk interpretasi model
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.model = model
        self.X_test = X_test
        self.shap_values = None
        self.logger = self._setup_logger()
        self.metrics = self.calculate_metrics()
        
    def _setup_logger(self):
        """Setup logger untuk evaluator"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Menghitung berbagai metrik evaluasi dengan penanganan kasus khusus
        
        Returns:
            Dict berisi berbagai metrik evaluasi
        """
        metrics = {
            'mse': mean_squared_error(self.y_true, self.y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            'mae': mean_absolute_error(self.y_true, self.y_pred),
            'r2': r2_score(self.y_true, self.y_pred),
            'explained_variance': explained_variance_score(self.y_true, self.y_pred),
            'max_error': max_error(self.y_true, self.y_pred),
            'mean_error': np.mean(self.y_true - self.y_pred),
            'std_error': np.std(self.y_true - self.y_pred)
        }
        
        # Hitung MAPE hanya jika tidak ada nilai 0 di y_true
        try:
            metrics['mape'] = mean_absolute_percentage_error(self.y_true, self.y_pred) * 100
        except ValueError:
            metrics['mape'] = np.inf
            self.logger.warning("Terdapat nilai 0 pada y_true, MAPE tidak dapat dihitung")
        
        # Hitung metrik tambahan jika model tersedia
        if self.model is not None and self.X_test is not None:
            try:
                self._calculate_shap_values()
                metrics['mean_abs_shap'] = np.mean(np.abs(self.shap_values))
            except Exception as e:
                self.logger.warning(f"Tidak dapat menghitung SHAP values: {str(e)}")
        
        return metrics
    
    def _calculate_shap_values(self, sample_size: int = 100):
        """Menghitung SHAP values untuk interpretasi model"""
        if self.model is None or self.X_test is None:
            raise ValueError("Model dan X_test diperlukan untuk menghitung SHAP values")
        
        # Sample data untuk efisiensi
        if len(self.X_test) > sample_size:
            sample_idx = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_sample = self.X_test[sample_idx]
        else:
            X_sample = self.X_test
        
        # Hitung SHAP values
        if hasattr(self.model, 'predict_proba'):
            explainer = shap.Explainer(self.model)
            self.shap_values = explainer(X_sample).values
        else:
            explainer = shap.Explainer(self.model)
            self.shap_values = explainer(X_sample).values
    
    def get_performance_report(self) -> str:
        """Membuat laporan performa model dalam format string"""
        report = "\n=== MODEL PERFORMANCE REPORT ===\n"
        for name, value in self.metrics.items():
            report += f"{name.upper():<20}: {value:.4f}\n"
        return report
    
    def plot_residuals(self, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        Visualisasi residual plot dengan informasi tambahan
        
        Parameters:
            save_path (optional): Path untuk menyimpan plot
            
        Returns:
            plt.Figure jika save_path tidak ditentukan
        """
        residuals = self.y_true - self.y_pred
        
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=self.y_pred, y=residuals, alpha=0.6)
        
        # Tambahkan garis referensi
        plt.axhline(y=0, color='r', linestyle='--')
        
        # Tambahkan garis rata-rata residual
        mean_residual = np.mean(residuals)
        plt.axhline(y=mean_residual, color='b', linestyle='-', 
                   label=f'Mean Residual: {mean_residual:.2f}')
        
        # Hitung dan plot interval kepercayaan
        std_residual = np.std(residuals)
        plt.axhline(y=mean_residual + 1.96*std_residual, color='g', linestyle=':',
                   label='95% Confidence Interval')
        plt.axhline(y=mean_residual - 1.96*std_residual, color='g', linestyle=':')
        
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Analysis')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Residual plot disimpan di: {save_path}")
        else:
            return plt
    
    def plot_actual_vs_predicted(self, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Visualisasi aktual vs prediksi dengan informasi tambahan"""
        plt.figure(figsize=(12, 8))
        
        # Scatter plot
        ax = sns.scatterplot(x=self.y_true, y=self.y_pred, alpha=0.6)
        
        # Garis diagonal
        min_val = min(self.y_true.min(), self.y_pred.min())
        max_val = max(self.y_true.max(), self.y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
        
        # Garis regresi
        coef = np.polyfit(self.y_true, self.y_pred, 1)
        poly1d_fn = np.poly1d(coef)
        plt.plot(self.y_true, poly1d_fn(self.y_true), 'b-', 
                label=f'Regression Line (slope={coef[0]:.2f})')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Actual vs Predicted plot disimpan di: {save_path}")
        else:
            return plt
    
    def plot_error_distribution(self, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Visualisasi distribusi error dengan informasi statistik"""
        errors = self.y_true - self.y_pred
        
        plt.figure(figsize=(12, 8))
        
        # Histogram dengan KDE
        ax = sns.histplot(errors, kde=True, bins=30)
        
        # Tambahkan garis statistik
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        plt.axvline(mean_error, color='r', linestyle='-', 
                   label=f'Mean Error: {mean_error:.2f}')
        plt.axvline(mean_error + std_error, color='g', linestyle='--',
                   label=f'±1 Std Dev: {std_error:.2f}')
        plt.axvline(mean_error - std_error, color='g', linestyle='--')
        
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Error distribution plot disimpan di: {save_path}")
        else:
            return plt
    
    def plot_shap_summary(self, feature_names: list = None, save_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Visualisasi SHAP summary plot"""
        if self.shap_values is None:
            self.logger.warning("SHAP values belum dihitung")
            return None
            
        plt.figure(figsize=(14, 8))
        shap.summary_plot(self.shap_values, self.X_test, feature_names=feature_names, show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            self.logger.info(f"SHAP summary plot disimpan di: {save_path}")
        else:
            return plt
    
    def save_evaluation_results(self, save_dir: str):
        """
        Menyimpan semua hasil evaluasi termasuk plot dan metrik
        
        Parameters:
            save_dir: Direktori untuk menyimpan hasil
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Simpan metrik
        with open(save_path / 'evaluation_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Simpan plot
        self.plot_residuals(save_path / 'residual_plot.png')
        self.plot_actual_vs_predicted(save_path / 'actual_vs_predicted.png')
        self.plot_error_distribution(save_path / 'error_distribution.png')
        
        # Simpan SHAP plot jika tersedia
        if self.shap_values is not None:
            self.plot_shap_summary(save_path=save_path / 'shap_summary.png')
        
        self.logger.info(f"Hasil evaluasi disimpan di: {save_path}")