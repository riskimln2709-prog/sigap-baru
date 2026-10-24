import polars as pl
import networkx as nx
import math

def calculate_haversine_polars(df: pl.DataFrame, lat2=5.5483, lon2=95.3238) -> pl.DataFrame:
    """Kalkulasi jarak Haversine geodesik secara native & tervektorisasi di Polars."""
    R = 6371.0  # Radius bumi (km)

    # Konversi derajat ke radian menggunakan ekspresi Polars
    lat1_rad = pl.col('lat_faskes').radians()
    lon1_rad = pl.col('lng_faskes').radians()
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dphi = lat2_rad - lat1_rad
    dlambda = lon2_rad - lon1_rad

    a = (
        (dphi / 2.0).sin().pow(2) +
        lat1_rad.cos() * math.cos(lat2_rad) * (dlambda / 2.0).sin().pow(2)
    )

    # Tambahkan kolom jarak_km
    return df.with_columns(
        (2.0 * R * a.sqrt().arcsin()).alias('jarak_km')
    )

def run_multi_tier_rules(df: pl.DataFrame) -> pl.DataFrame:
    """Menjalankan 3 Lapisan Screening Rules Engine menggunakan Polars."""
    
    # Hitung jarak jika kolom koordinat tersedia
    if 'lat_faskes' in df.columns and 'lng_faskes' in df.columns:
        df = calculate_haversine_polars(df)

    # 1. Lapis 1 (Hard Rules): Status Kartu Non-Aktif
    rule_kartu = pl.when(pl.col('status_aktif') == 0).then(60.0).otherwise(0.0)

    # 2. Lapis 2 (Spatiotemporal Anomaly): Jarak >= 300km dalam waktu <= 3 jam
    rule_spatio = pl.when(
        (pl.col('jarak_km') >= 300.0) & (pl.col('waktu_selisih_jam') <= 3.0)
    ).then(75.0).otherwise(0.0)

    # 3. Hitung Skor Maksimum Aturan
    df = df.with_columns(
        pl.max_horizontal([rule_kartu, rule_spatio]).alias('skor_rules')
    )

    # 4. Generate Explainable Evidence Trail (XAI Text)
    df = df.with_columns(
        pl.when(pl.col('skor_rules') >= 60.0)
        .then(pl.lit("🚨 Peringatan: Anomali Kartu Non-Aktif / Impossible Travel"))
        .when(pl.col('skor_rules') >= 35.0)
        .then(pl.lit("⚠️ Review: Indikasi Berisiko Ringan"))
        .otherwise(pl.lit("✅ Klaim Normal / Wajar"))
        .alias('evidence_trail')
    )

    return df

if __name__ == "__main__":
    print("Mesin Dewa berbasis Polars siap digunakan.")