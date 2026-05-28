import os
import pandas as pd
from config.database import get_engine


# ── Konfigurasi ───────────────────────────────────────────────────────────────
RAW_DIR    = "data/raw"
CLEAN_PATH = "data/data_bersih.csv"
TABLE_NAME = "poverty_panel_data"

PROVINCE_FIXES = {
    "D.I. YOGYAKARTA"   : "DI YOGYAKARTA",
    "D.I.Y. YOGYAKARTA" : "DI YOGYAKARTA",
    "DIY YOGYAKARTA"    : "DI YOGYAKARTA",
    "D.K.I. JAKARTA"    : "DKI JAKARTA",
    "D K I JAKARTA"     : "DKI JAKARTA",
    "PAPIUA"            : "PAPUA",
}


# ── Helper ────────────────────────────────────────────────────────────────────
def standarisasi_provinsi(nama: str) -> str:
    nama = str(nama).strip().upper()
    return PROVINCE_FIXES.get(nama, nama)


def wide_to_long(df: pd.DataFrame, nama_kolom: str) -> pd.DataFrame:
    """Ubah kolom tahun (2015, 2016, ...) menjadi baris."""
    kolom_tahun = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]
    df_long = df.melt(id_vars="Provinsi", value_vars=kolom_tahun,
                      var_name="Tahun", value_name=nama_kolom)
    df_long["Tahun"]     = df_long["Tahun"].astype(int)
    df_long[nama_kolom]  = pd.to_numeric(df_long[nama_kolom], errors="coerce")
    return df_long


# ── E : Extract ───────────────────────────────────────────────────────────────
def extract(raw_dir: str) -> dict[str, pd.DataFrame]:
    """Baca semua file CSV dari folder raw."""
    return {
        f.removesuffix(".csv"): pd.read_csv(os.path.join(raw_dir, f))
        for f in os.listdir(raw_dir)
        if f.endswith(".csv")
    }


# ── T : Transform ─────────────────────────────────────────────────────────────
def transform(raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Bersihkan dan gabungkan dataset RLS, TPT, dan Kemiskinan."""

    rls_list, tpt_list, kemiskinan_list = [], [], []

    for nama_file, df in raw_data.items():
        # Standarisasi kolom
        df.columns  = df.columns.str.strip()
        df          = df.rename(columns={"provinsi": "Provinsi"})
        df["Provinsi"] = df["Provinsi"].apply(standarisasi_provinsi)

        # Kelompokkan berdasarkan nama file
        key = nama_file.lower()
        if "rls" in key or "sekolah" in key:
            rls_list.append(wide_to_long(df, "Rata_rata_Lama_Sekolah"))
        elif "tpt" in key or "pengangguran" in key:
            tpt_list.append(wide_to_long(df, "Tingkat_Pengangguran_Terbuka"))
        elif "miskin" in key or "kemiskinan" in key:
            kemiskinan_list.append(wide_to_long(df, "Persentase_Kemiskinan"))

    def gabung_dan_agregasi(daftar, kolom):
        """Concat lalu rata-ratakan duplikat per Provinsi–Tahun."""
        if not daftar:
            return pd.DataFrame(columns=["Provinsi", "Tahun", kolom])
        return (pd.concat(daftar)
                  .groupby(["Provinsi", "Tahun"], as_index=False)
                  .mean())

    df_rls       = gabung_dan_agregasi(rls_list,       "Rata_rata_Lama_Sekolah")
    df_tpt       = gabung_dan_agregasi(tpt_list,       "Tingkat_Pengangguran_Terbuka")
    df_kemiskinan = gabung_dan_agregasi(kemiskinan_list, "Persentase_Kemiskinan")

    # Gabungkan tiga dataset
    panel = (df_rls
             .merge(df_tpt,        on=["Provinsi", "Tahun"], how="outer")
             .merge(df_kemiskinan, on=["Provinsi", "Tahun"], how="outer"))

    return (panel
            .dropna(subset=["Provinsi", "Tahun"])
            .sort_values(["Provinsi", "Tahun"])
            .reset_index(drop=True))


# ── L : Load ──────────────────────────────────────────────────────────────────
def load(df: pd.DataFrame):
    """Simpan data bersih ke CSV lokal dan upload ke PostgreSQL Aiven."""
    # Backup CSV
    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"[+] CSV tersimpan → {CLEAN_PATH} ({len(df)} baris)")

    # Upload ke Aiven
    df.to_sql(TABLE_NAME, con=get_engine(), if_exists="replace",
              index=False, method="multi")
    print(f"[+] Upload selesai → tabel '{TABLE_NAME}' di Aiven")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Memulai Pipeline ETL ===\n")

    print("1. Extract — membaca file CSV...")
    raw_data = extract(RAW_DIR)

    if not raw_data:
        print(f"[!] Tidak ada file CSV di folder '{RAW_DIR}'")
    else:
        print("2. Transform — membersihkan & menggabungkan data...")
        df_bersih = transform(raw_data)

        print("3. Load — menyimpan data...")
        load(df_bersih)

    print("\n=== Pipeline ETL Selesai ===")