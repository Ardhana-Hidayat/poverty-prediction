import os
import sys
import pandas as pd

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from config.database import get_engine

# konfigurasi
RAW_DATA_DIR = "data/clean"
OUTPUT_CSV_PATH = "data/dataset_cleaned.csv"
DATABASE_TABLE_NAME = "poverty_panel_data"

DATASET_FILES = {
    "education": "rata_rata_lama_sekolah.csv",
    "unemployment": "tingkat_pengangguran.csv",
    "poverty": "persentase_penduduk_miskin.csv"
}

# EXTRACT
def extract_data() -> dict[str, pd.DataFrame]:
    print("EXTRACT")

    extracted_data = {}

    for dataset_name, filename in DATASET_FILES.items():
        file_path = os.path.join(RAW_DATA_DIR, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

        df = pd.read_csv(file_path)

        extracted_data[dataset_name] = df

        print(
            f"[OK] {filename:<35} "
            f"{df.shape[0]} baris × {df.shape[1]} kolom"
        )

    return extracted_data


# TRANSFORM
def transform_data(extracted_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("\nTRANSFORM")

    education_df = extracted_data["education"]
    unemployment_df = extracted_data["unemployment"]
    poverty_df = extracted_data["poverty"]

    # Merge dataset
    merged_df = education_df.merge(
        unemployment_df,
        on=["Provinsi", "Tahun"],
        how="inner"
    )

    merged_df = merged_df.merge(
        poverty_df,
        on=["Provinsi", "Tahun"],
        how="inner"
    )

    # Rename kolom
    COLUMN_RENAME_MAP = {
        "RLS": "Rata_Rata_Lama_Sekolah",
        "TPT_Agustus": "Tingkat_Pengangguran_Terbuka",
        "Persentase_Kemiskinan": "Persentase_Penduduk_Miskin"
    }

    merged_df = merged_df.rename(columns=COLUMN_RENAME_MAP)

    # Sort data
    merged_df = (
        merged_df
        .sort_values(by=["Provinsi", "Tahun"])
        .reset_index(drop=True)
    )

    print(
        f"[OK] Dataset gabungan berhasil dibuat "
        f"({merged_df.shape[0]} baris × {merged_df.shape[1]} kolom)"
    )

    return merged_df

# LOAD
def load_data(final_df: pd.DataFrame) -> None:
    print("\nLOAD")

    # Simpan backup lokal
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

    final_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"[OK] Backup CSV tersimpan → {OUTPUT_CSV_PATH}")

    # Upload ke database Aiven
    try:
        engine = get_engine()

        final_df.to_sql(
            DATABASE_TABLE_NAME,
            con=engine,
            if_exists="replace",
            index=False,
            method="multi"
        )

        print(
            f"[OK] Database berhasil diperbarui "
            f"→ tabel '{DATABASE_TABLE_NAME}'"
        )

    except Exception as error:
        print(f"[GAGAL] Upload database gagal: {error}")


# MAIN
if __name__ == "__main__":
    print("--> Analisis Data Sosial Ekonomi Indonesia <--\n")

    extracted_data = extract_data()

    transformed_data = transform_data(extracted_data)

    load_data(transformed_data)

    print("\n--> Pipeline ETL Selesai <--")