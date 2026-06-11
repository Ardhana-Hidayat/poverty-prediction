import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config.database import get_engine

# konfigurasi
CLEAN_DATA_DIR      = "data/processed"
OUTPUT_CSV_PATH     = "data/dataset_cleaned.csv"
DATABASE_TABLE_NAME = "poverty_panel_data"

DATASET_FILES = {
    "education"    : "rata_rata_lama_sekolah.csv",
    "unemployment" : "tingkat_pengangguran.csv",
    "poverty"      : "persentase_penduduk_miskin.csv",
}

COLUMN_RENAME_MAP = {
    "RLS"                   : "Rata_Rata_Lama_Sekolah",
    "TPT_Agustus"           : "Tingkat_Pengangguran_Terbuka",
    "Persentase_Kemiskinan" : "Persentase_Penduduk_Miskin",
}

# ── Extract ───────────────────────────────────────────────────────────────────
def extract() -> dict[str, pd.DataFrame]:
    print("EXTRACT:")
    data = {}
    for name, filename in DATASET_FILES.items():
        path = os.path.join(CLEAN_DATA_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File tidak ditemukan: {path}")
        df = pd.read_csv(path)
        data[name] = df
        print(f"  [OK] {filename:<35} {df.shape[0]} baris × {df.shape[1]} kolom")
    return data

# ── Transform ─────────────────────────────────────────────────────────────────
def transform(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("\nTRANSFORM:")

    merged = data["education"].merge(
        data["unemployment"], on=["Provinsi", "Tahun"], how="inner"
    ).merge(
        data["poverty"], on=["Provinsi", "Tahun"], how="inner"
    )

    merged = merged.rename(columns=COLUMN_RENAME_MAP)
    merged = merged.sort_values(["Provinsi", "Tahun"]).reset_index(drop=True)

    print(
        f"  [OK] Dataset gabungan: "
        f"{merged.shape[0]} baris × {merged.shape[1]} kolom"
    )
    return merged

# ── Load ──────────────────────────────────────────────────────────────────────
def load(df: pd.DataFrame) -> None:
    print("\nLOAD")

    # Simpan CSV lokal
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"  [OK] CSV tersimpan → {OUTPUT_CSV_PATH}")

    # Upload ke Aiven PostgreSQL
    try:
        engine = get_engine()
        df.to_sql(
            DATABASE_TABLE_NAME,
            con=engine,
            if_exists="replace",
            index=False,
            method="multi",
        )
        print(f"  [OK] Database diperbarui → tabel '{DATABASE_TABLE_NAME}'")
    except Exception as e:
        print(f"  [GAGAL] Upload database gagal: {e}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Pipeline ETL\n")
    load(transform(extract()))
    print("\nPipeline Selesai")