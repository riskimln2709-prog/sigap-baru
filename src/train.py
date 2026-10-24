import polars as pl
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def train_risk_model(data_path, model_output_path="models/risk_model.pkl"):
    """Melatih model XGBoost untuk memprediksi risiko klaim (Fraud / Normal)."""
    if not os.path.exists(data_path):
        print(f"Dataset tidak ditemukan di {data_path}. Jalankan preprocessing terlebih dahulu.")
        return
        
    df = pl.read_csv(data_path)
    
    if 'is_risky' not in df.columns:
        print("Kolom target 'is_risky' tidak ditemukan dalam dataset.")
        return
        
    # Memisahkan fitur dan target (abaikan kolom hash/identitas)
    X = df.drop(columns=['is_risky', 'nik_hash', 'bpjs_hash'], errors='ignore')
    X = X.select_dtypes(include=['number']) # Hanya ambil fitur numerik untuk contoh
    y = df['is_risky']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inisialisasi dan latih model XGBoost
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model berhasil dilatih! Akurasi uji: {acc * 100:.2f}%")
    
    # Simpan model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model berhasil disimpan di {model_output_path}")

if __name__ == "__main__":
    # Contoh jalur dataset hasil olahan
    train_risk_model("data/processed/claims_processed.csv")