import os
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.database import get_engine

# ── Konfigurasi ───────────────────────────────────────────────────────────────
RAW_DIR    = "data/raw"
CLEAN_PATH = "data/dataset_cleaned.csv"
TABLE_NAME = "poverty_panel_data"

PROVINCE_FIXES = {
    "D.I. YOGYAKARTA"   : "DI YOGYAKARTA",
    "D.I.Y. YOGYAKARTA" : "DI YOGYAKARTA",
    "DIY YOGYAKARTA"    : "DI YOGYAKARTA",
    "D.K.I. JAKARTA"    : "DKI JAKARTA",
    "D K I JAKARTA"     : "DKI JAKARTA",
    "PAPIUA"            : "PAPUA",
}

PAPUA_DOB = [
    "PAPUA BARAT DAYA",
    "PAPUA PEGUNUNGAN",
    "PAPUA SELATAN",
    "PAPUA TENGAH",
]


# ── Helper ────────────────────────────────────────────────────────────────────
def standarisasi_provinsi(nama: str) -> str:
    nama = str(nama).strip().upper()
    return PROVINCE_FIXES.get(nama, nama)


# ── Tahap 1: Extract ──────────────────────────────────────────────────────────
def run_extract() -> dict[str, pd.DataFrame]:
    print("── EXTRACT ──────────────────────────────────────────")
    raw_data = {
        f.removesuffix(".csv"): pd.read_csv(os.path.join(RAW_DIR, f))
        for f in os.listdir(RAW_DIR)
        if f.endswith(".csv")
    }

    assert raw_data, f"[FAIL] Tidak ada file CSV di '{RAW_DIR}'"
    print(f"[OK]   {len(raw_data)} file ditemukan: {list(raw_data.keys())}")

    for nama, df in raw_data.items():
        assert not df.empty, f"[FAIL] File '{nama}' kosong"
        print(f"[OK]   {nama}: {df.shape[0]} baris × {df.shape[1]} kolom")

    return raw_data


# ── Tahap 2: Transform ────────────────────────────────────────────────────────
def run_transform(raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("\n── TRANSFORM ────────────────────────────────────────")

    df_rls, df_tpt, df_kemiskinan = None, None, None

    for nama_file, df in raw_data.items():
        # Standarisasi kolom & provinsi
        df.columns     = df.columns.str.strip()
        df              = df.rename(columns={"provinsi": "Provinsi"})
        df["Provinsi"] = df["Provinsi"].apply(standarisasi_provinsi)
        df["Tahun"]    = df["Tahun"].astype(int)

        key = nama_file.lower()

        if "rls" in key or "sekolah" in key:
            # Sudah long format — rename kolom saja
            df_rls = df[["Provinsi", "Tahun", "RLS"]].rename(
                columns={"RLS": "Rata_rata_Lama_Sekolah"}
            )

        elif "tpt" in key or "pengangguran" in key:
            # Rata-ratakan 2 periode (Februari + Agustus)
            df["Tingkat_Pengangguran_Terbuka"] = df[["TPT_Februari", "TPT_Agustus"]].mean(axis=1)
            df_tpt = df[["Provinsi", "Tahun", "Tingkat_Pengangguran_Terbuka"]]

        elif "miskin" in key or "kemiskinan" in key:
            # Rata-ratakan 2 semester (Maret + September)
            df["Persentase_Kemiskinan"] = df[["Jumlah_S1_Maret", "Jumlah_S2_Sept"]].mean(axis=1)
            df_kemiskinan = df[["Provinsi", "Tahun", "Persentase_Kemiskinan"]]

    # Gabungkan tiga dataset
    panel = (df_rls
             .merge(df_tpt,        on=["Provinsi", "Tahun"], how="outer")
             .merge(df_kemiskinan, on=["Provinsi", "Tahun"], how="outer"))

    df = (panel
          .dropna(subset=["Provinsi", "Tahun"])
          .sort_values(["Provinsi", "Tahun"])
          .reset_index(drop=True))

    assert not df.empty,                                   "[FAIL] Hasil transform kosong"
    assert not df.duplicated(["Provinsi", "Tahun"]).any(), "[FAIL] Duplikat Provinsi–Tahun ditemukan"
    assert df["Tahun"].between(2000, 2030).all(),          "[FAIL] Tahun di luar rentang 2000–2030"

    print(f"[OK]   Shape    : {df.shape[0]} baris × {df.shape[1]} kolom")
    print(f"[OK]   Tahun    : {df['Tahun'].min()} – {df['Tahun'].max()}")
    print(f"[OK]   Provinsi : {df['Provinsi'].nunique()} unik, tidak ada duplikat")

    return df


# ── Tahap 3: Drop Papua DOB ───────────────────────────────────────────────────
def run_drop_papua_dob(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── DROP PAPUA DOB ───────────────────────────────────")

    sebelum = len(df)
    df      = df[~df["Provinsi"].isin(PAPUA_DOB)].reset_index(drop=True)
    sesudah = len(df)

    print(f"[OK]   {sebelum - sesudah} baris dihapus ({', '.join(PAPUA_DOB)})")
    print(f"[OK]   Sisa: {sesudah} baris, {df['Provinsi'].nunique()} provinsi")

    return df


# ── Tahap 4: Load ─────────────────────────────────────────────────────────────
def run_load(df: pd.DataFrame):
    print("\n── LOAD ─────────────────────────────────────────────")

    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"[OK]   CSV tersimpan  → {CLEAN_PATH} ({len(df)} baris)")

    df.to_sql(TABLE_NAME, con=get_engine(), if_exists="replace",
              index=False, method="multi")
    print(f"[OK]   Aiven tersimpan → tabel '{TABLE_NAME}'")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Pipeline ETL — Analisis Sosial Ekonomi Indonesia ===\n")

    raw_data = run_extract()
    df       = run_transform(raw_data)
    df       = run_drop_papua_dob(df)
    run_load(df)

    print("\n=== Pipeline selesai ✅ ===")