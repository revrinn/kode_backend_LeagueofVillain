# backend/src/train_performance_predictor.py
import numpy as np
import sys
import os
import json
import logging
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Setup paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from performance_prediction.data_processor import PerformanceDataProcessor
from performance_prediction.model_trainer import PerformanceModelTrainer
from performance_prediction.evaluator import PerformanceEvaluator

def main():
    try:
        logger.info("=== MEMULAI PELATIHAN MODEL PREDIKSI PERFORMA ===")
        
        # Setup paths
        BASE_DIR = current_dir.parent.parent
        DATA_PATH = BASE_DIR / "backend/data/processed/cleaned_education_data.csv"  # Ensure this is the correct path
        MODEL_SAVE_DIR = BASE_DIR / "models/performance_predictor/trained_model"
        LOG_DIR = BASE_DIR / "models/performance_predictor/training_logs"
        CONFIG_PATH = BASE_DIR / "config/model_config.json"

        # Buat direktori jika belum ada
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # 1. Persiapan Data
        logger.info("\n=== MEMUAT DAN MEMPROSES DATA ===")
        processor = PerformanceDataProcessor(DATA_PATH, CONFIG_PATH)
        data = processor.load_data()
        
        # Cek data
        if data is None or data.empty:
            logger.error("Data kosong atau gagal dimuat")
            return
            
        # Siapkan fitur dan target
        features, target = processor.prepare_features_target(data)
        
        # Bagi data menjadi train, validation, dan test set
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
            test_size=0.2,
            val_size=0.2
        )
        
        # Gunakan subset data untuk testing jika perlu
        # X_train, y_train = X_train[:1000], y_train[:1000]
        # X_val, y_val = X_val[:1000], y_val[:1000]
        
        # 2. Pelatihan Model
        logger.info("\n=== MELATIH MODEL ===")
        trainer = PerformanceModelTrainer()
        
        # Gunakan parameter yang lebih konservatif untuk testing
        best_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 1,
            'gamma': 0
        }
        
        # Latih model final dengan parameter
        model = trainer.train_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params=best_params
        )
        
        # 3. Evaluasi Model
        logger.info("\n=== EVALUASI MODEL ===")
        evaluation = trainer.evaluate_model(X_test, y_test)
        
        logger.info("\n=== DETAIL EVALUASI ===")
        logger.info(f"Contoh 5 prediksi pertama: {evaluation['predictions'][:5]}")
        logger.info(f"Contoh 5 nilai sebenarnya: {y_test[:5]}")
        logger.info(f"Perbedaan prediksi dan aktual: {np.abs(y_test[:5] - evaluation['predictions'][:5])}")

        # Simpan metrik evaluasi
        metrics = evaluation['metrics']
        with open(MODEL_SAVE_DIR / "model_metrics.json", 'w') as f:json.dump(metrics, f, indent=4)
        
        # Visualisasi evaluasi
        evaluator = PerformanceEvaluator(y_test, evaluation['predictions'])
        
        # Plot dan simpan visualisasi
        plots = {
            "residual_plot": evaluator.plot_residuals(),
            "actual_vs_predicted": evaluator.plot_actual_vs_predicted(),
            "error_distribution": evaluator.plot_error_distribution()
        }
        
        for name, plot in plots.items():
            plot_path = LOG_DIR / f"{name}.png"
            plot.savefig(plot_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Plot {name} disimpan di: {plot_path}")
        
        # Plot dari model trainer
        trainer.plot_learning_curve(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            save_path=LOG_DIR / "learning_curve.png"
        )

        feature_plot, importance_df = trainer.plot_feature_importance(
            feature_names=processor.features.columns,
            save_path=LOG_DIR / "feature_importance.png"
        )
        
        # Simpan feature importance
        importance_df.to_csv(LOG_DIR / "feature_importance.csv", index=False)
        
        # SHAP summary plot
        try:
            trainer.plot_shap_summary(
                feature_names=processor.features.columns,
                save_path=LOG_DIR / "shap_summary.png"
            )
        except Exception as e:
            logger.warning(f"Tidak dapat membuat SHAP plot: {str(e)}")
        
        # 4. Simpan Model dan Processor
        logger.info("\n=== MENYIMPAN MODEL ===")
        model_path = trainer.save_model(MODEL_SAVE_DIR)
        processor_path = processor.save_processor(MODEL_SAVE_DIR)
        
        logger.info("\n=== PELATIHAN SELESAI ===")
        logger.info(f"Model disimpan di: {model_path}")
        logger.info(f"Processor disimpan di: {processor_path}")
        print(f"Log dan visualisasi disimpan di: {LOG_DIR}")

    except Exception as e:
        logger.error(f"Terjadi kesalahan saat melatih model: {str(e)}")

if __name__ == "__main__":
    main()
