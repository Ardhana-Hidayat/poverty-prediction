"""
model_klasifikasi.py
====================
Model Klasifikasi — Prediksi Kategori Risiko Kemiskinan (>15%)
Dataset: 34 provinsi × 2015–2025

Dua model dibandingkan:
  1. Logistic Regression — baseline interpretable
  2. Random Forest       — model utama (non-linear, ensemble)

Keputusan desain:
  - Evaluasi: Walk-forward validation (bukan KFold biasa) untuk data panel
    Walk-forward: train pada tahun < t, test pada tahun t (t=2020–2025)
  - Target: Biner (Risiko Tinggi jika Persentase Penduduk Miskin > 15.0%)
  - Imbalanced class (18% data) ditangani menggunakan class_weight='balanced'
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Konfigurasi ───────────────────────────────────────────────────────────────
CLEAN_DATA_PATH = "data/dataset_cleaned.csv"
MODEL_DIR       = "model/klasifikasi"

PAPUA_DOB = [
    "PAPUA BARAT DAYA", "PAPUA PEGUNUNGAN",
    "PAPUA SELATAN",    "PAPUA TENGAH",
]
STRUCTURAL_BREAK = {"provinsi": "PAPUA", "tahun_min": 2024}

FEATURES = [
    "Rata_Rata_Lama_Sekolah",
    "Tingkat_Pengangguran_Terbuka",
    "RLS_lag1",
    "TPT_lag1",
    "RLS_YoY",
    "TPT_YoY",
    "RLS_x_TPT",
    "Tahun_norm",
]
TARGET_REGRESS = "Persentase_Penduduk_Miskin"
TARGET_CLASS   = "Risiko_Tinggi"
THRESHOLD      = 15.0
RANDOM_STATE   = 42

# Walk-forward: train <t, test =t, mulai 2020
TAHUN_TEST_START = 2020


# ── 1. Load & Clean ───────────────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    print("── 1. LOAD & CLEAN ──────────────────────────────────")
    df = pd.read_csv(path)
    n0 = len(df)

    mask_dob = df["Provinsi"].isin(PAPUA_DOB)
    mask_sb  = (
        (df["Provinsi"] == STRUCTURAL_BREAK["provinsi"]) &
        (df["Tahun"]    >= STRUCTURAL_BREAK["tahun_min"])
    )
    df = df[~mask_dob & ~mask_sb].reset_index(drop=True)
    print(f"[OK] {n0} → {len(df)} baris "
          f"({n0 - len(df)} dihapus: Papua DOB + structural break)")
    return df


# ── 2. Feature Engineering ────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n── 2. FEATURE ENGINEERING & TARGET CREATION ─────────")
    df = df.sort_values(["Provinsi", "Tahun"]).copy()

    # Buat fitur temporal & interaksi
    df["RLS_lag1"]   = df.groupby("Provinsi")["Rata_Rata_Lama_Sekolah"].shift(1)
    df["TPT_lag1"]   = df.groupby("Provinsi")["Tingkat_Pengangguran_Terbuka"].shift(1)
    df["RLS_YoY"]    = df.groupby("Provinsi")["Rata_Rata_Lama_Sekolah"].pct_change() * 100
    df["TPT_YoY"]    = df.groupby("Provinsi")["Tingkat_Pengangguran_Terbuka"].pct_change() * 100
    df["RLS_x_TPT"]  = df["Rata_Rata_Lama_Sekolah"] * df["Tingkat_Pengangguran_Terbuka"]
    df["Tahun_norm"] = (df["Tahun"] - 2015) / 10

    # Buat target klasifikasi
    df[TARGET_CLASS] = (df[TARGET_REGRESS] > THRESHOLD).astype(int)

    df_feat = df.dropna(subset=FEATURES + [TARGET_CLASS]).reset_index(drop=True)
    
    n_total = len(df_feat)
    n_positive = df_feat[TARGET_CLASS].sum()
    prop_positive = (n_positive / n_total) * 100
    
    print(f"[OK] Fitur: {FEATURES}")
    print(f"[OK] Target Biner: '{TARGET_CLASS}' (Kemiskinan > {THRESHOLD}%)")
    print(f"[OK] Distribusi Target: Risiko Tinggi = {n_positive}/{n_total} ({prop_positive:.2f}%)")
    print(f"[OK] Baris setelah rekayasa: {n_total} "
          f"(berkurang {len(df) - n_total} karena lag tahun pertama)")
    return df_feat


# ── 3. Walk-forward Validation ────────────────────────────────────────────────
def walk_forward_validate(df: pd.DataFrame) -> dict:
    print("\n── 3. WALK-FORWARD VALIDATION ───────────────────────")
    print(f"     Skema: train tahun < t, test tahun t "
          f"(t={TAHUN_TEST_START}–{df['Tahun'].max()})")
    print(f"     Penanganan Imbalance: class_weight='balanced'\n")

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest"      : RandomForestClassifier(
            n_estimators=200, max_depth=6,
            min_samples_leaf=3, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
    }

    tahun_list = sorted(df[df["Tahun"] >= TAHUN_TEST_START]["Tahun"].unique())
    cv_results = {}

    for name, model in models.items():
        print(f"=== Evaluasi Model: {name} ===")
        print(f"{'Tahun':>8} {'N':>4} {'Positif':>8} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC AUC':>10}")
        print("-" * 79)

        all_y_true, all_y_pred, all_y_prob = [], [], []
        
        for t in tahun_list:
            train = df[df["Tahun"] <  t]
            test  = df[df["Tahun"] == t]
            if len(test) == 0:
                continue

            model.fit(train[FEATURES], train[TARGET_CLASS])
            
            y_pred = model.predict(test[FEATURES])
            # Dapatkan probabilitas kelas positif untuk ROC AUC
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(test[FEATURES])[:, 1]
            else:
                y_prob = model.decision_function(test[FEATURES])

            y_true = test[TARGET_CLASS].values
            
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            all_y_prob.extend(y_prob)

            # Hitung metrik per-tahun
            acc = accuracy_score(y_true, y_pred)
            # handle zero division warnings if any class is missing in test year
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = np.nan # Terjadi jika hanya ada satu kelas dalam data uji tahun tersebut

            print(
                f"{t:>8} {len(test):>4} {y_true.sum():>8} "
                f"{acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {f"{auc:.4f}" if not np.isnan(auc) else "N/A":>10}"
            )

        # Hitung metrik agregat di seluruh periode walk-forward
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        all_y_prob = np.array(all_y_prob)

        overall_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
        overall_prec = precision_score(all_y_true, all_y_pred, zero_division=0)
        overall_rec = recall_score(all_y_true, all_y_pred, zero_division=0)
        overall_acc = accuracy_score(all_y_true, all_y_pred)
        overall_auc = roc_auc_score(all_y_true, all_y_prob)

        print("-" * 79)
        print(
            f"{'RATA-RATA':<13} {len(all_y_true):>4} {all_y_true.sum():>8} "
            f"{overall_acc:>10.4f} {overall_prec:>10.4f} {overall_rec:>10.4f} {overall_f1:>10.4f} {overall_auc:>10.4f}\n"
        )

        cv_results[name] = {
            "f1": overall_f1,
            "precision": overall_prec,
            "recall": overall_rec,
            "accuracy": overall_acc,
            "auc": overall_auc,
            "y_true": all_y_true,
            "y_pred": all_y_pred,
            "y_prob": all_y_prob,
            "model": model
        }

    best_name = max(cv_results, key=lambda k: cv_results[k]["f1"])
    print(f"[OK] Model terbaik berdasarkan F1-Score: {best_name} "
          f"(F1-Score={cv_results[best_name]['f1']:.4f}, ROC AUC={cv_results[best_name]['auc']:.4f})")
    
    return cv_results, best_name, tahun_list


# ── 4. Final Fit & Feature Importance ─────────────────────────────────────────
def final_fit(df: pd.DataFrame, cv_results: dict, best_name: str):
    print(f"\n── 4. FINAL FIT ({best_name}) ──────────────────────")
    best_model = cv_results[best_name]["model"]

    # Final fit menggunakan seluruh data yang tersedia untuk disave
    best_model.fit(df[FEATURES], df[TARGET_CLASS])
    print(f"[OK] Model terbaik dilatih kembali menggunakan seluruh dataset ({len(df)} baris).")

    # Ambil Feature Importance
    if "Random Forest" in best_name:
        clf = best_model
        importances = clf.feature_importances_
        print("\n[OK] Feature Importance (Random Forest):")
        imp = pd.Series(importances, index=FEATURES).sort_values(ascending=False)
        for feat, val in imp.items():
            bar = "█" * int(val * 40)
            print(f"     {feat:<35} {val:.4f}  {bar}")
    elif "Logistic Regression" in best_name:
        coefs = best_model.named_steps["model"].coef_[0]
        print("\n[OK] Coefficients (Logistic Regression):")
        imp = pd.Series(coefs, index=FEATURES).sort_values(key=abs, ascending=False)
        for feat, val in imp.items():
            sign = "+" if val >= 0 else "-"
            bar = "█" * int(abs(val) * 10)
            print(f"     {feat:<35} {val:>7.4f} ({sign}) {bar}")
            
    return best_model


# ── 5. Simpan Model ───────────────────────────────────────────────────────────
def save_model(model, path: str):
    print(f"\n── 5. SIMPAN MODEL ──────────────────────────────────")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[OK] Model klasifikasi tersimpan → {path}")


# ── 6. Visualisasi ────────────────────────────────────────────────────────────
def plot_results(cv_results: dict, best_name: str):
    print(f"\n── 6. VISUALISASI EVALUASI ──────────────────────────")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Evaluasi Model Klasifikasi — Prediksi Kategori Risiko Kemiskinan\n"
                 "(Walk-forward validation, target: kemiskinan > 15.0%)", fontsize=13)

    # Palette warna kustom
    colors = {"RF": "#378ADD", "LR": "#1D9E75", "Neutral": "#888888"}

    # Plot 1: Confusion Matrix untuk Model Terbaik
    ax = axes[0]
    best_results = cv_results[best_name]
    cm = confusion_matrix(best_results["y_true"], best_results["y_pred"])
    
    # Heatmap custom
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                annot_kws={"size": 12, "weight": "bold"})
    ax.set_title(f"Confusion Matrix ({best_name})", fontsize=11, pad=10)
    ax.set_xlabel("Prediksi", labelpad=8)
    ax.set_ylabel("Aktual", labelpad=8)
    ax.set_xticklabels(["Bukan Risiko Tinggi (0)", "Risiko Tinggi (1)"])
    ax.set_yticklabels(["Bukan Risiko Tinggi (0)", "Risiko Tinggi (1)"], rotation=0)

    # Plot 2: ROC Curve & PR Curve
    ax = axes[1]
    for name, res in cv_results.items():
        color = colors["RF"] if "Forest" in name else colors["LR"]
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", color=color, linewidth=1.8)
        
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Acak (AUC=0.500)")
    ax.set_xlabel("False Positive Rate", labelpad=8)
    ax.set_ylabel("True Positive Rate (Recall)", labelpad=8)
    ax.set_title("Kurva ROC (Receiver Operating Characteristic)", fontsize=11, pad=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Plot 3: Feature Importance (Random Forest) / Coefficients (Logistic Regression)
    ax = axes[2]
    model_repr = cv_results[best_name]["model"]
    
    if "Random Forest" in best_name:
        importances = model_repr.feature_importances_
        imp_series = pd.Series(importances, index=FEATURES).sort_values(ascending=True)
        ax.barh(imp_series.index, imp_series.values, color="#378ADD", alpha=0.85, edgecolor="white", height=0.6)
        ax.set_title(f"Feature Importance ({best_name})", fontsize=11, pad=10)
        ax.set_xlabel("Skor Kepentingan")
    else:
        coefs = model_repr.named_steps["model"].coef_[0]
        coef_series = pd.Series(coefs, index=FEATURES).sort_values(key=abs, ascending=True)
        # Warna hijau untuk koef positif, merah/biru untuk negatif
        bar_colors = ["#378ADD" if c >= 0 else "#D9534F" for c in coef_series.values]
        ax.barh(coef_series.index, coef_series.values, color=bar_colors, alpha=0.85, edgecolor="white", height=0.6)
        ax.set_title(f"Koefisien Model ({best_name})", fontsize=11, pad=10)
        ax.set_xlabel("Nilai Koefisien")

    plt.tight_layout()
    out_path = os.path.join(MODEL_DIR, "evaluasi_klasifikasi.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Plot evaluasi tersimpan → {out_path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Model Klasifikasi — Prediksi Risiko Kemiskinan ===\n")

    # Pastikan direktori model ada
    os.makedirs(MODEL_DIR, exist_ok=True)

    df       = load_and_clean(CLEAN_DATA_PATH)
    df_feat  = build_features(df)

    cv_results, best_name, tahun_list = walk_forward_validate(df_feat)
    best_model                        = final_fit(df_feat, cv_results, best_name)

    model_suffix = "rf" if "Random Forest" in best_name else "lr"
    model_path = os.path.join(MODEL_DIR, f"model_klasifikasi_{model_suffix}.pkl")
    save_model(best_model, model_path)
    plot_results(cv_results, best_name)

    print("\n=== Selesai ✅ ===")
