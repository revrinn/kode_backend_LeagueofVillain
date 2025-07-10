# backend/src/performance_prediction/model_trainer.py

import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import joblib
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
import logging
from functools import partial
import shap
import random

class PerformanceModelTrainer:
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.shap_values = None
        self.best_params = None
        self.cv_results = None
        self.logger = self._setup_logger()
        self.study = None
        
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def objective(self, trial, X, y):
        """Fungsi objective untuk Optuna dengan error handling yang lebih baik"""
        try:
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),  # Diperbarui range
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42,
                'n_jobs': 1  
            }
            
            model = xgb.XGBRegressor(**params)
            
            # Gunakan KFold cross-validation dengan error handling
            kf = KFold(n_splits=3, shuffle=True, random_state=42)  # Kurangi splits untuk efisiensi
            
            try:
                scores = cross_val_score(
                    model, X, y, 
                    cv=kf, 
                    scoring='neg_mean_squared_error',
                    n_jobs=1,  
                    error_score='raise'
                )
                return np.mean(scores)
            except Exception as e:
                self.logger.warning(f"Trial gagal: {str(e)}")
                return float('-inf')  # Return nilai terburuk jika gagal
                
        except Exception as e:
            self.logger.error(f"Error dalam objective function: {str(e)}")
            return float('-inf')

        
    def hyperparameter_tuning(self, X_train, y_train, n_trials=30):
        """Alternatif sederhana jika Optuna bermasalah"""
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        
        best_score = float('-inf')
        best_params = {}
        
        for _ in range(n_trials):
            params = {k: random.choice(v) for k, v in param_grid.items()}
            # Hapus n_estimators untuk xgb.train
            train_params = params.copy()
            train_params.pop('n_estimators', None)
            
            model = xgb.XGBRegressor(**params, random_state=42)
            score = cross_val_score(model, X_train, y_train, 
                                cv=3, scoring='neg_mean_squared_error').mean()
            
            if score > best_score:
                best_score = score
                best_params = params
        
        self.best_params = best_params
        return best_params

    def train_model(self, X_train, y_train, X_val=None, y_val=None, params=None):
        """Melatih model final dengan early stopping"""
        try:
            self.logger.info("\n=== TRAINING FINAL MODEL ===")
            
            if params is None and self.best_params is not None:
                params = self.best_params
            
            # Parameter default
            default_params = {
                'objective': 'reg:squarederror',
                'random_state': 42,
                'verbosity': 1
            }
            
            # Hapus n_estimators jika menggunakan xgb.train
            if 'n_estimators' in params:
                params.pop('n_estimators')
                
            final_params = {**default_params, **(params or {})}
            
            if X_val is not None and y_val is not None:
                self.logger.info("Menggunakan early stopping dengan validation set")
                
                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_val, label=y_val)
                
                evals = [(dtrain, 'train'), (dval, 'val')]
                evals_result = {}
                model = xgb.train(
                    final_params,
                    dtrain,
                    num_boost_round=1000,
                    evals=evals,
                    early_stopping_rounds=50,
                    verbose_eval=50,
                    evals_result=evals_result
                )
                
                # Simpan evals_result
                self.evals_result = evals_result
            else:
                self.logger.info("Training tanpa early stopping")
                model = xgb.XGBRegressor(**final_params)
                model.fit(X_train, y_train)
            
            self.model = model
            
            # Hitung feature importance dan SHAP values
            self._calculate_feature_importance(X_train)
            self._calculate_shap_values(X_train)
            
            return model
        except Exception as e:
            self.logger.error(f"Error dalam training model: {str(e)}")
            raise

    def _calculate_feature_importance(self, X_train):
        """Menghitung feature importance"""
        try:
            if isinstance(self.model, xgb.Booster):
                # Untuk model Booster (xgb.train)
                importance = self.model.get_score(importance_type='weight')
                # Konversi ke format yang konsisten
                self.feature_importance = {k: float(v) for k, v in importance.items()}
            elif hasattr(self.model, 'feature_importances_'):
                # Untuk model scikit-learn API (XGBRegressor)
                self.feature_importance = dict(zip(
                    self.model.get_booster().feature_names,
                    self.model.feature_importances_
                ))
            else:
                self.logger.warning("Tipe model tidak dikenali untuk menghitung feature importance")
                self.feature_importance = None
        except Exception as e:
            self.logger.error(f"Gagal menghitung feature importance: {str(e)}")
            self.feature_importance = None
            
    def _calculate_shap_values(self, X_train, sample_size=100):
        """Menghitung SHAP values untuk interpretasi model"""
        try:
            if self.model is None:
                raise ValueError("Model belum dilatih")
                
            if isinstance(self.model, xgb.Booster):
                explainer = shap.TreeExplainer(self.model)
                X_sample = shap.utils.sample(X_train, sample_size)
                self.shap_values = explainer.shap_values(X_sample)
            else:
                explainer = shap.Explainer(self.model)
                self.shap_values = explainer(X_train)
        except Exception as e:
            self.logger.warning(f"Tidak dapat menghitung SHAP values: {str(e)}")
            self.shap_values = None
            
    def evaluate_model(self, X_test, y_test):
        """Evaluasi model dengan metrik lengkap"""
        try:
            if self.model is None:
                raise ValueError("Model belum dilatih")
            
            dtest = xgb.DMatrix(X_test)
            predictions = self.model.predict(dtest)
            
            # Hitung berbagai metrik evaluasi
            metrics = self._calculate_all_metrics(y_test, predictions)
            
            self.logger.info("\n=== HASIL EVALUASI MODEL ===")
            for name, value in metrics.items():
                self.logger.info(f"{name}: {value:.4f}")
            
            return {
                'metrics': metrics,
                'predictions': predictions,
                'shap_values': self.shap_values
            }
            
        except Exception as e:
            self.logger.error(f"Error dalam evaluasi model: {str(e)}")
            raise
    
    def _calculate_all_metrics(self, y_true, y_pred):
        """Menghitung semua metrik evaluasi"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'max_error': np.max(np.abs(y_true - y_pred))
        }
        
        # Hitung MAPE dengan penanganan nilai 0
        try:
            # Tambahkan epsilon kecil untuk menghindari division by zero
            y_true_adjusted = np.where(y_true == 0, 1e-10, y_true)
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true_adjusted)) * 100
        except Exception as e:
            metrics['mape'] = np.inf
            self.logger.warning(f"Tidak dapat menghitung MAPE: {str(e)}")
        
        return metrics
    
    def plot_learning_curve(self, X_train, y_train, X_val, y_val, save_path=None):
        """Visualisasi learning curve"""
        try:
            # Gunakan evals_result yang sudah disimpan
            if not hasattr(self, 'evals_result') or not self.evals_result:
                self.logger.warning("Tidak ada evals_result tersedia untuk learning curve")
                return None
            
            results = self.evals_result
            epochs = len(results['train']['rmse']) if 'train' in results else 0
            
            if epochs == 0:
                self.logger.warning("Data learning curve kosong")
                return None
                
            x_axis = range(0, epochs)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(x_axis, results['train']['rmse'], label='Train')
            
            if 'val' in results:
                ax.plot(x_axis, results['val']['rmse'], label='Validation')
                
            ax.legend()
            plt.ylabel('RMSE')
            plt.xlabel('Epochs')
            plt.title('XGBoost Learning Curve')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Learning curve disimpan di: {save_path}")
            else:
                return plt
                
        except Exception as e:
            self.logger.error(f"Error membuat learning curve: {str(e)}")
            raise
    
    def plot_feature_importance(self, feature_names=None, top_n=20, save_path=None):
        """Visualisasi feature importance"""
        try:
            if self.feature_importance is None:
                self._calculate_feature_importance(feature_names)  # Coba hitung lagi
                
            if self.feature_importance is None:
                raise ValueError("Feature importance belum dihitung. Model mungkin belum dilatih atau terjadi error dalam perhitungan.")
            
            # Buat DataFrame dari feature importance
            importance_df = pd.DataFrame({
                'feature': list(self.feature_importance.keys()),
                'importance': list(self.feature_importance.values())
            }).sort_values('importance', ascending=False)
            
            # Jika ada feature_names, pastikan urutannya benar
            if feature_names is not None:
                importance_df = importance_df[importance_df['feature'].isin(feature_names)]
            
            # Ambil top N features
            top_features = importance_df.head(top_n)
            
            # Plot
            plt.figure(figsize=(14, 10))
            bars = plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('Importance Score')
            plt.title('Top Feature Importance')
            
            # Tambahkan nilai importance
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{width:.4f}',
                        va='center', ha='left')
            
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Feature importance plot disimpan di: {save_path}")
                return None, importance_df
            else:
                return plt, importance_df
                
        except Exception as e:
            self.logger.error(f"Error membuat feature importance plot: {str(e)}")
            raise
        
    
    def plot_shap_summary(self, feature_names=None, save_path=None):
        """Visualisasi SHAP summary plot"""
        try:
            if self.shap_values is None:
                raise ValueError("SHAP values belum dihitung")
            
            plt.figure(figsize=(14, 10))
            shap.summary_plot(self.shap_values, feature_names=feature_names, show=False)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()
                self.logger.info(f"SHAP summary plot disimpan di: {save_path}")
            else:
                return plt
                
        except Exception as e:
            self.logger.error(f"Error membuat SHAP summary plot: {str(e)}")
            raise
    
    def save_model(self, save_dir, model_name=None):
        """Menyimpan model dan semua hasil terkait"""
        try:
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if not model_name:
                model_name = f"performance_model_{timestamp}"
            
            # Path untuk berbagai file
            model_path = Path(save_dir) / f"{model_name}.pkl"
            params_path = Path(save_dir) / f"{model_name}_params.json"
            cv_path = Path(save_dir) / f"{model_name}_cv_results.csv"
            shap_path = Path(save_dir) / f"{model_name}_shap_values.npy"
            study_path = Path(save_dir) / f"{model_name}_optuna_study.pkl"
            
            # Simpan model
            joblib.dump(self.model, model_path)
            
            # Simpan parameter terbaik
            with open(params_path, 'w') as f:
                json.dump(self.best_params, f, indent=4)
            
            # Simpan hasil CV jika ada
            if self.cv_results is not None:
                pd.DataFrame(self.cv_results).to_csv(cv_path, index=False)
            
            # Simpan SHAP values jika ada
            if self.shap_values is not None:
                np.save(shap_path, self.shap_values, allow_pickle=True)
            
            # Simpan optuna study jika ada
            if self.study is not None:
                joblib.dump(self.study, study_path)
            
            self.logger.info("\n=== MODEL DISIMPAN ===")
            self.logger.info(f"Model: {model_path}")
            self.logger.info(f"Parameter: {params_path}")
            if self.cv_results is not None:
                self.logger.info(f"Hasil CV: {cv_path}")
            if self.shap_values is not None:
                self.logger.info(f"SHAP values: {shap_path}")
            if self.study is not None:
                self.logger.info(f"Optuna study: {study_path}")
            
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"Error menyimpan model: {str(e)}")
            raise