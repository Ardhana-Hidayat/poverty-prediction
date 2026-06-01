"""
model_clustering.py
======================
Model Clustering — Segmentasi Provinsi Berdasarkan Indikator Sosial-Ekonomi
Dataset: 34 provinsi (Diambil data tahun terakhir untuk kondisi terkini)

Pendekatan model:
  1. K-Means Clustering — baseline untuk partitional clustering
  2. Evaluasi menggunakan Silhouette Score & Elbow Method (Inertia)

Keputusan desain:
  - Karena ini data panel, kita agregasi mengambil tahun terakhir (kondisi paling update)
    untuk setiap provinsi agar segmentasi merepresentasikan kondisi saat ini.
  - Data di-scaling menggunakan StandardScaler karena unit indikator berbeda
    (Persentase vs Tahun/Lama Sekolah).
  - Algoritma otomatis mencari K terbaik berdasarkan nilai Silhouette Score tertinggi.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Konfigurasi ───────────────────────────────────────────────────────────────
CLEAN_DATA_PATH = "data/dataset_cleaned.csv"
MODEL_DIR       = "model/clustering"
MODEL_PATH      = os.path.join(MODEL_DIR, "model_clustering_kmeans.pkl")

PAPUA_DOB = [
    "PAPUA BARAT DAYA", "PAPUA PEGUNUNGAN",
    "PAPUA SELATAN",    "PAPUA TENGAH",
]
STRUCTURAL_BREAK = {"provinsi": "PAPUA", "tahun_min": 2024}

# Fitur utama yang digunakan untuk clustering (tanpa lag/turunan)
FEATURES = [
    "Persentase_Penduduk_Miskin",
    "Rata_Rata_Lama_Sekolah",
    "Tingkat_Pengangguran_Terbuka"
]
RANDOM_STATE = 42
K_MIN, K_MAX = 2, 6


# ── 1. Load & Clean ───────────────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    print("── 1. LOAD & CLEAN ──────────────────────────────────")
    df = pd.read_csv(path)
    n0 = len(df)

    # Filter yang sama dengan model lain agar konsisten
    mask_dob = df["Provinsi"].isin(PAPUA_DOB)
    mask_sb  = (
        (df["Provinsi"] == STRUCTURAL_BREAK["provinsi"]) &
        (df["Tahun"]    >= STRUCTURAL_BREAK["tahun_min"])
    )
    df = df[~mask_dob & ~mask_sb].reset_index(drop=True)
    print(f"[OK] {n0} → {len(df)} baris "
          f"({n0 - len(df)} dihapus: Papua DOB + structural break)")
    return df


# ── 2. Feature Engineering (Agregasi) ─────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── 2. FEATURE ENGINEERING (AGREGASI) ────────────────")
    # Ambil data tahun terakhir untuk setiap provinsi
    df = df.sort_values(["Provinsi", "Tahun"])
    df_latest = df.groupby("Provinsi").last().reset_index()
    
    df_feat = df_latest[["Provinsi", "Tahun"] + FEATURES].dropna().reset_index(drop=True)
    print(f"[OK] Fitur Clustering: {FEATURES}")
    print(f"[OK] Total Provinsi yang disegmentasi: {len(df_feat)} (Tahun: {df_feat['Tahun'].max()})")
    return df_feat


# ── 3. Evaluasi K-Means & Silhouette ──────────────────────────────────────────
def evaluate_clusters(df: pd.DataFrame) -> dict:
    print("\n── 3. EVALUASI JUMLAH CLUSTER (K) ───────────────────")
    
    X = df[FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}
    print(f"{'K':<5} {'Inertia':>12} {'Silhouette Score':>18}")
    print("-" * 38)

    for k in range(K_MIN, K_MAX + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        inertia = kmeans.inertia_
        sil_score = silhouette_score(X_scaled, labels)
        
        results[k] = {
            "model": kmeans,
            "labels": labels,
            "inertia": inertia,
            "silhouette": sil_score
        }
        print(f"{k:<5} {inertia:>12.4f} {sil_score:>18.4f}")

    # Pilih K dengan Silhouette terbaik
    best_k = max(results, key=lambda k: results[k]["silhouette"])
    print(f"\n[OK] K terbaik terpilih: {best_k} "
          f"(Silhouette={results[best_k]['silhouette']:.4f})")
    
    return results, best_k, scaler, X_scaled


# ── 4. Final Fit & Analisis Cluster ───────────────────────────────────────────
def final_fit(df: pd.DataFrame, results: dict, best_k: int, scaler) -> KMeans:
    print(f"\n── 4. ANALISIS KARAKTERISTIK CLUSTER (K={best_k}) ───────")
    
    best_model = results[best_k]["model"]
    df_final = df.copy()
    df_final["Cluster"] = best_model.labels_

    # Print rata-rata tiap fitur per cluster untuk interpretasi
    cluster_means = df_final.groupby("Cluster")[FEATURES].mean()
    cluster_counts = df_final["Cluster"].value_counts().sort_index()

    for c in range(best_k):
        print(f"\nCluster {c} (Total: {cluster_counts[c]} Provinsi):")
        for feat in FEATURES:
            print(f"    {feat:<30}: {cluster_means.loc[c, feat]:.2f}")
            
    # Menyimpan nama provinsi per cluster untuk diintip
    print("\n[OK] Distribusi Provinsi:")
    for c in range(best_k):
        prov_list = df_final[df_final["Cluster"] == c]["Provinsi"].tolist()
        print(f"  C{c}: {', '.join(prov_list[:5])}{'...' if len(prov_list)>5 else ''}")

    # Gabungkan scaler dan model untuk disimpan
    model_pipeline = {
        "scaler": scaler,
        "model": best_model,
        "features": FEATURES
    }
    
    return model_pipeline, df_final


# ── 5. Simpan Model ───────────────────────────────────────────────────────────
def save_model(model_data, path: str):
    print(f"\n── 5. SIMPAN MODEL ──────────────────────────────────")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"[OK] Model tersimpan → {path}")


# ── 6. Visualisasi ────────────────────────────────────────────────────────────
def plot_results(df_final: pd.DataFrame, results: dict, best_k: int):
    print(f"\n── 6. VISUALISASI ───────────────────────────────────")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Evaluasi Model Clustering — K-Means (K={best_k})\nSegmentasi Kondisi Sosial-Ekonomi Terkini", fontsize=13)

    # Plot 1: Elbow Method (Inertia & Silhouette)
    ax1 = axes[0]
    k_vals = list(results.keys())
    inertias = [results[k]["inertia"] for k in k_vals]
    silhouettes = [results[k]["silhouette"] for k in k_vals]

    color1 = "#378ADD"
    ax1.set_xlabel("Jumlah Cluster (K)")
    ax1.set_ylabel("Inertia", color=color1)
    ax1.plot(k_vals, inertias, marker="o", color=color1, label="Inertia")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#1D9E75"
    ax2.set_ylabel("Silhouette Score", color=color2)
    ax2.plot(k_vals, silhouettes, marker="s", color=color2, linestyle="--", label="Silhouette")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax1.set_title("Evaluasi Metrik (Elbow & Silhouette)")

    # Plot 2: Scatter Plot Kemiskinan vs Lama Sekolah
    ax3 = axes[1]
    scatter = ax3.scatter(
        df_final["Persentase_Penduduk_Miskin"], 
        df_final["Rata_Rata_Lama_Sekolah"], 
        c=df_final["Cluster"], 
        cmap="Set1", alpha=0.7, edgecolors="white", s=60
    )
    ax3.set_xlabel("Persentase Penduduk Miskin (%)")
    ax3.set_ylabel("Rata-rata Lama Sekolah (Tahun)")
    ax3.set_title("Sebaran Cluster Provinsi")
    
    # Bikin legenda otomatis
    legend_labels = [f"Cluster {i}" for i in range(best_k)]
    handles, _ = scatter.legend_elements()
    ax3.legend(handles, legend_labels, title="Segmen")

    plt.tight_layout()
    out_path = os.path.join(MODEL_DIR, "evaluasi_clustering.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Plot tersimpan → {out_path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Model Clustering — Segmentasi Provinsi ===\n")

    df        = load_and_clean(CLEAN_DATA_PATH)
    df_feat   = build_features(df)

    results, best_k, scaler, X_scaled = evaluate_clusters(df_feat)
    model_pipeline, df_final          = final_fit(df_feat, results, best_k, scaler)

    save_model(model_pipeline, MODEL_PATH)
    plot_results(df_final, results, best_k)

    print("\n=== Selesai ✅ ===")