import polars as pl
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
import os

def train_hybrid_models():
    print("⚡ Memuat dataset menggunakan Polars Engine...")
    data_path = "data/raw/claims_synthetic_10k.csv"
    
    # Read CSV via Polars
    df = pl.read_csv(data_path)
    
    # Feature columns
    feature_cols = ['status_aktif', 'jarak_km', 'waktu_selisih_jam', 'biaya_juta', 'lama_rawat', 'jumlah_tindakan']
    
    # Konversi Polars DataFrame ke NumPy Array untuk XGBoost & Isolation Forest
    X = df.select(feature_cols).to_numpy()
    
    # Target label sintetis (Risk Flag)
    y = df.select('is_fraud').to_numpy().ravel()

    print("🤖 Melatih Model XGBoost (Supervised 70%)...")
    xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    xgb_model.fit(X, y)

    print("🌲 Melatih Model Isolation Forest (Unsupervised 30%)...")
    iforest_model = IsolationForest(contamination=0.05, random_state=42)
    iforest_model.fit(X)

    # Simpan Model
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb_model, "models/model_sigap_xgb_v2.pkl")
    joblib.dump(iforest_model, "models/model_sigap_iforest_v2.pkl")

    print("✅ Model AI berbasis Polars berhasil dilatih dan disimpan di folder 'models/'!")

if __name__ == "__main__":
    train_hybrid_models()