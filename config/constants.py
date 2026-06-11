import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CSV_PATH   = os.path.join(PROJECT_ROOT, "data", "dataset_cleaned.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "model_regresi_rf.pkl")

DATABASE_TABLE = "poverty_panel_data"

# definisikan features
FEATURES = [
    "Rata_Rata_Lama_Sekolah",
    "Tingkat_Pengangguran_Terbuka",
    "Tahun",
]

# target = variabel yang mau diprediksi
TARGET = "Persentase_Penduduk_Miskin"