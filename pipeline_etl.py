import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.constants import CSV_PATH, DATABASE_TABLE
from config.database  import get_engine

# konfigurasi
CLEAN_DATA_DIR = "data/processed"

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

# extract
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

# transform
def transform(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("\nTRANSFORM:")

    merged = (
        data["education"]
        .merge(data["unemployment"], on=["Provinsi", "Tahun"], how="inner")
        .merge(data["poverty"],      on=["Provinsi", "Tahun"], how="inner")
        .rename(columns=COLUMN_RENAME_MAP)
        .sort_values(["Provinsi", "Tahun"])
        .reset_index(drop=True)
    )

    print(f"  [OK] Dataset gabungan: {merged.shape[0]} baris × {merged.shape[1]} kolom")
    return merged

# load
def load(df: pd.DataFrame) -> None:
    print("LOAD:")

    df.to_csv(CSV_PATH, index=False)
    print(f"  [OK] CSV tersimpan → {CSV_PATH}")

    try:
        engine = get_engine()
        df.to_sql(DATABASE_TABLE, con=engine, if_exists="replace", index=False, method="multi")
        print(f"  [OK] Database diperbarui → tabel '{DATABASE_TABLE}'")
    except Exception as e:
        print(f"  [GAGAL] Upload database gagal: {e}")

# main
if __name__ == "__main__":
    print("Pipeline ETL:\n")
    load(transform(extract()))
    print("\nSelesai")