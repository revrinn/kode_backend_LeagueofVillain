# backend/src/recommendation/utils.py
from pathlib import Path
import pandas as pd
import joblib
import os

def load_data(data_path=None):
    """
    Memuat data yang sudah diproses dengan path yang lebih fleksibel
    """
    if data_path is None:
        # Cari file di beberapa lokasi yang mungkin
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        possible_paths = [
            os.path.join(base_dir, 'backend/data/processed/cleaned_education_data.csv'),  # Dari root project
            os.path.join(base_dir, 'data/processed/cleaned_education_data.csv'),         # Alternatif
            'data/processed/cleaned_education_data.csv',                                 # Relatif
            '../data/processed/cleaned_education_data.csv'                              # Dari src
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                print(f"Data ditemukan di: {data_path}")
                break
        else:
            raise FileNotFoundError(
                "Tidak dapat menemukan file data. Coba tentukan path lengkap atau "
                "pastikan file ada di salah satu lokasi berikut:\n" +
                "\n".join(possible_paths))
    
    # Pastikan path menggunakan separator yang benar untuk OS
    data_path = os.path.normpath(data_path)
    
    try:
        df = pd.read_csv(data_path)
        print(f"Data berhasil dimuat dari: {data_path}")
        return df
    except Exception as e:
        raise Exception(f"Gagal memuat data dari {data_path}: {str(e)}")

def save_evaluation_results(results, model_name, save_dir='data/recommendations/evaluations'):
    """
    Menyimpan hasil evaluasi model
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame([results])
    results_df['model'] = model_name
    
    save_path = os.path.join(save_dir, f"{model_name}_evaluation.csv")
    results_df.to_csv(save_path, index=False)
    
    print(f"Hasil evaluasi untuk {model_name} disimpan di {save_path}")

def get_user_history(df, user_id):
    """
    Mendapatkan riwayat materi yang diakses oleh user tertentu
    """
    user_data = df[df['student_id'] == user_id]
    return user_data['material_type_encoded'].tolist()