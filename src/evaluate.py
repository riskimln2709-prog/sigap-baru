import polars as pl
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def evaluate_model(model_path="models/risk_model.pkl", test_data_path="data/processed/claims_processed.csv"):
    """Mengevaluasi performa model Machine Learning SIGAP."""
    if not os.path.exists(model_path) or not os.path.exists(test_data_path):
        print("Model atau file data pengujian tidak ditemukan.")
        return
        
    model = joblib.load(model_path)
    df = pl.read_csv(test_data_path)
    
    if 'is_risky' not in df.columns:
        print("Kolom target 'is_risky' tidak ditemukan.")
        return
        
    X = df.drop(columns=['is_risky', 'nik_hash', 'nama_hash'], errors='ignore')
    X = X.select_dtypes(include=['number'])
    y = df['is_risky']
    
    preds = model.predict(X)
    
    print("========================================")
    print("      LAPORAN EVALUASI MODEL SIGAP      ")
    print("========================================")
    print(f"Akurasi Keseluruhan: {accuracy_score(y, preds) * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, preds))
    print("\nClassification Report:")
    print(classification_report(y, preds))

if __name__ == "__main__":
    evaluate_model()