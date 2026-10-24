import polars as pl
import numpy as np
import random
import os

def generate_sigap_10k_data(output_path="data/raw/claims_synthetic_10k.csv", n_samples=10000):
    np.random.seed(42)
    random.seed(42)
    
    # Koordinat Faskes Utama (Contoh: Aceh & Jakarta)
    lat_aceh, lng_aceh = 5.5483, 95.3238
    lat_jkt, lng_jkt = -6.2088, 106.8456
    
    data = []
    # Kumpulan No HP Sindikat (dipakai bersama oleh banyak NIK)
    sindikat_hps = [f"0812{random.randint(10000000, 99999999)}" for _ in range(5)]

    for i in range(n_samples):
        nik = f"117101{random.randint(1000000000, 9999999999)}"
        no_bpjs = f"000{random.randint(1000000000, 9999999999)}"
        
        # 1. Hard Rule: Status Aktif (5% Non-Aktif)
        status_aktif = 0 if random.random() < 0.05 else 1
        
        # 2. Graph Rule: No HP (3% Sindikat)
        is_sindikat = random.random() < 0.03
        no_hp = random.choice(sindikat_hps) if is_sindikat else f"0812{random.randint(10000000, 99999999)}"
        
        # 3. Spatiotemporal: Jarak & Waktu
        is_impossible_travel = random.random() < 0.04
        if is_impossible_travel:
            lat_faskes = lat_jkt + np.random.normal(0, 0.01)
            lng_faskes = lng_jkt + np.random.normal(0, 0.01)
            waktu_selisih_jam = random.uniform(0.5, 2.5) # Jarak jauh tapi waktu cepat
        else:
            lat_faskes = lat_aceh + np.random.normal(0, 0.05)
            lng_faskes = lng_aceh + np.random.normal(0, 0.05)
            waktu_selisih_jam = random.uniform(12.0, 72.0)

        # Hitung kasar jarak dari Aceh ke Faskes
        jarak_km = 1700.0 if is_impossible_travel else random.uniform(2.0, 45.0)
        
        # Fitur Finansial
        lama_rawat = random.randint(1, 14)
        jumlah_tindakan = random.randint(1, 12)
        biaya_juta = (lama_rawat * 0.45) + (jumlah_tindakan * 0.3) + random.uniform(0.2, 1.0)
        
        # Label Fraud Sebenarnya (Ground Truth untuk Evaluasi Akurasi 94%)
        is_fraud = 1 if (status_aktif == 0 or is_sindikat or is_impossible_travel or biaya_juta > 8.0) else 0

        data.append({
            'nik': nik,
            'no_bpjs': no_bpjs,
            'no_hp': no_hp,
            'status_aktif': status_aktif,
            'lat_faskes': lat_faskes,
            'lng_faskes': lng_faskes,
            'jarak_km': jarak_km,
            'waktu_selisih_jam': waktu_selisih_jam,
            'lama_rawat': lama_rawat,
            'jumlah_tindakan': jumlah_tindakan,
            'biaya_juta': biaya_juta,
            'is_fraud': is_fraud
        })

    df = pl.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset 10.000 Klaim Sintetis SIGAP Berhasil Dibuat: '{output_path}'")

if __name__ == "__main__":
    generate_sigap_10k_data()