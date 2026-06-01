import os
import pandas as pd

# ── Konfigurasi ───────────────────────────────────────────────────────────────
RAW_DATA_DIR   = "data/raw"
CLEAN_DATA_DIR = "data/clean"

DATASET_FILES = {
    "education"    : "rata_rata_lama_sekolah.csv",
    "unemployment" : "tingkat_pengangguran.csv",
    "poverty"      : "persentase_penduduk_miskin.csv",
}

# Provinsi DOB Papua 2022 — seluruh nilai merupakan backfill, bukan observasi nyata
PAPUA_DOB = [
    "PAPUA BARAT DAYA",
    "PAPUA PEGUNUNGAN",
    "PAPUA SELATAN",
    "PAPUA TENGAH",
]

# Papua induk 2024–2025 — structural break akibat redistribusi penduduk ke DOB
# RLS loncat +3.36 tahun dalam satu tahun, tidak merepresentasikan kondisi nyata
STRUCTURAL_BREAK = {"provinsi": "PAPUA", "tahun_min": 2024}


# ── Helper ────────────────────────────────────────────────────────────────────
def drop_papua(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Menghapus dua kategori baris bermasalah terkait Papua:
      1. Papua DOB      — 4 provinsi pemekaran 2022, data backfill
      2. Structural break — Papua induk 2024+ akibat redistribusi penduduk DOB

    Parameters
    ----------
    df    : DataFrame hasil baca CSV raw, wajib punya kolom 'Provinsi' & 'Tahun'
    label : nama dataset untuk keperluan log (misal "RLS", "TPT", "Kemiskinan")

    Returns
    -------
    pd.DataFrame bersih tanpa baris Papua bermasalah
    """
    total_awal = len(df)

    # 1. Papua DOB
    mask_dob = df["Provinsi"].isin(PAPUA_DOB)
    n_dob    = mask_dob.sum()
    for provinsi in PAPUA_DOB:
        n = (df["Provinsi"] == provinsi).sum()
        if n > 0:
            print(f"  [DROP] {provinsi:<25} {n:>2} baris  (DOB Papua 2022)")

    # 2. Structural break Papua 2024–2025
    mask_sb = (
        (df["Provinsi"] == STRUCTURAL_BREAK["provinsi"]) &
        (df["Tahun"]    >= STRUCTURAL_BREAK["tahun_min"])
    )
    n_sb = mask_sb.sum()
    for _, row in df[mask_sb].iterrows():
        print(
            f"  [DROP] {row['Provinsi']:<25} {int(row['Tahun'])}  "
            f"(structural break Papua)"
        )

    # Eksekusi
    df_clean = df[~mask_dob & ~mask_sb].reset_index(drop=True)

    print(
        f"  [OK]  {label}: {total_awal} → {len(df_clean)} baris "
        f"({n_dob + n_sb} dihapus: {n_dob} DOB + {n_sb} structural break)\n"
    )

    return df_clean


def validate(df: pd.DataFrame, label: str) -> None:
    """Pastikan tidak ada sisa baris Papua bermasalah."""
    assert df["Provinsi"].isin(PAPUA_DOB).sum() == 0, \
        f"[{label}] Papua DOB masih tersisa!"
    assert len(df[
        (df["Provinsi"] == STRUCTURAL_BREAK["provinsi"]) &
        (df["Tahun"]    >= STRUCTURAL_BREAK["tahun_min"])
    ]) == 0, f"[{label}] Structural break Papua masih tersisa!"
    assert df.duplicated(["Provinsi", "Tahun"]).sum() == 0, \
        f"[{label}] Duplikat Provinsi-Tahun ditemukan!"
    assert df["Provinsi"].notna().all(), \
        f"[{label}] Ada nilai kosong di kolom Provinsi!"

    print(f"  [OK]  Validasi {label} lulus")


# ── Cleaner per dataset ───────────────────────────────────────────────────────
def clean_education(df: pd.DataFrame) -> pd.DataFrame:
    print("── RLS (Rata-rata Lama Sekolah) ─────────────────────")
    df = drop_papua(df, "RLS")
    validate(df, "RLS")
    return df


def clean_unemployment(df: pd.DataFrame) -> pd.DataFrame:
    print("── TPT (Tingkat Pengangguran Terbuka) ───────────────")
    df = drop_papua(df, "TPT")
    validate(df, "TPT")
    return df


def clean_poverty(df: pd.DataFrame) -> pd.DataFrame:
    print("── Kemiskinan (Persentase Penduduk Miskin) ──────────")
    df = drop_papua(df, "Kemiskinan")
    validate(df, "Kemiskinan")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("--> Pipeline Cleaner — Data Papua <--\n")

    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)

    cleaners = {
        "education"    : clean_education,
        "unemployment" : clean_unemployment,
        "poverty"      : clean_poverty,
    }

    for dataset_name, cleaner_fn in cleaners.items():
        filename  = DATASET_FILES[dataset_name]
        raw_path  = os.path.join(RAW_DATA_DIR, filename)
        clean_path = os.path.join(CLEAN_DATA_DIR, filename)

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"File tidak ditemukan: {raw_path}")

        df_raw   = pd.read_csv(raw_path)
        df_clean = cleaner_fn(df_raw)
        df_clean.to_csv(clean_path, index=False)

        print(f"  [SAVE] {clean_path}\n")

    print("--> Pipeline Cleaner Selesai <--")