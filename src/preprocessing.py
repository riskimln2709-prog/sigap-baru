import polars as pl
import hashlib

SALT = "SIGAP_BPJS_2026_SECRET_SALT"

def hash_value(val):
    """Melakukan Salted SHA-256 Hashing untuk anonimisasi NIK dan No. BPJS sesuai UU PDP."""
    if val is None or val == "":
        return ""
    salted_str = f"{SALT}_{val}"
    return hashlib.sha256(salted_str.encode('utf-8')).hexdigest()[:16]

def load_and_preprocess_data(file_path):
    """Memuat data klaim, menganonimkan NIK & No. BPJS (Purification Pipeline), dan membersihkan null values."""
    try:
        # Reset pointer file jika input berupa Streamlit UploadedFile / BytesIO
        if hasattr(file_path, 'seek'):
            file_path.seek(0)
            df = pl.read_csv(file_path.read())
        else:
            df = pl.read_csv(file_path)
    except Exception as e:
        print(f"File tidak dapat dibaca oleh Polars: {e}")
        return None

    # 1. Purification Pipeline (UU PDP): Hash NIK -> nik_hash
    if 'nik' in df.columns:
        df = df.with_columns(
            pl.col('nik').cast(pl.Utf8).map_elements(hash_value, return_dtype=pl.Utf8).alias('nik_hash')
        ).drop('nik')

    # 2. Purification Pipeline (UU PDP): Hash No. BPJS -> bpjs_hash
    if 'no_bpjs' in df.columns:
        df = df.with_columns(
            pl.col('no_bpjs').cast(pl.Utf8).map_elements(hash_value, return_dtype=pl.Utf8).alias('bpjs_hash')
        ).drop('no_bpjs')

    # Fallback jika masih ada dataset lama berkolom 'nama'
    if 'nama' in df.columns:
        df = df.drop('nama')

    # Cleaning missing values (Polars)
    df = df.fill_null(0)
    return df

if __name__ == "__main__":
    print("Modul preprocessing SIGAP berbasis Polars (No. BPJS) siap digunakan.")