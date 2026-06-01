"""
03_model_regresi.py
===================
Model Regresi — Prediksi Persentase Kemiskinan
Dataset: 34 provinsi × 2015–2025 (372 → 338 baris setelah feature engineering)

Tiga model dibandingkan:
  1. Ridge Regression    — baseline interpretable
  2. Random Forest       — model utama (non-linear, ensemble)
  3. Gradient Boosting   — pembanding boosting

Keputusan desain:
  - Evaluasi: Walk-forward validation (bukan KFold biasa)
    KFold acak menyebabkan data leakage temporal pada data panel.
    Walk-forward: train pada tahun < t, test pada tahun t (t=2020–2025)
  - Target di-log-transform (skewness=1.133)
  - Miskin_lag1 TIDAK dipakai sebagai fitur (r=0.997, trivially predictive)
  - Fitur turunan: lag, YoY%, interaksi RLS×TPT, tren waktu
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Konfigurasi ───────────────────────────────────────────────────────────────
CLEAN_DATA_PATH = "data/dataset_cleaned.csv"
MODEL_DIR       = "model"
MODEL_PATH      = os.path.join(MODEL_DIR, "model_regresi_rf.pkl")

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
TARGET       = "Persentase_Penduduk_Miskin"
RANDOM_STATE = 42

# Walk-forward: train <t, test =t, mulai 2020 (4 tahun training awal)
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
    print("\n── 2. FEATURE ENGINEERING ───────────────────────────")
    df = df.sort_values(["Provinsi", "Tahun"]).copy()

    df["RLS_lag1"]   = df.groupby("Provinsi")["Rata_Rata_Lama_Sekolah"].shift(1)
    df["TPT_lag1"]   = df.groupby("Provinsi")["Tingkat_Pengangguran_Terbuka"].shift(1)
    df["RLS_YoY"]    = df.groupby("Provinsi")["Rata_Rata_Lama_Sekolah"].pct_change() * 100
    df["TPT_YoY"]    = df.groupby("Provinsi")["Tingkat_Pengangguran_Terbuka"].pct_change() * 100
    df["RLS_x_TPT"]  = df["Rata_Rata_Lama_Sekolah"] * df["Tingkat_Pengangguran_Terbuka"]
    df["Tahun_norm"] = (df["Tahun"] - 2015) / 10

    df_feat = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    print(f"[OK] Fitur: {FEATURES}")
    print(f"[OK] Baris: {len(df_feat)} "
          f"(berkurang {len(df) - len(df_feat)} karena lag tahun pertama)")
    print("[NOTE] Miskin_lag1 dikecualikan (r=0.997, trivially predictive)")
    return df_feat


# ── 3. Walk-forward Validation ────────────────────────────────────────────────
def walk_forward_validate(df: pd.DataFrame) -> dict:
    print("\n── 3. WALK-FORWARD VALIDATION ───────────────────────")
    print(f"     Skema: train tahun < t, test tahun t "
          f"(t={TAHUN_TEST_START}–{df['Tahun'].max()})")
    print(f"     Target: log1p(kemiskinan) → evaluasi dibalik ke skala asli\n")

    models = {
        "Ridge"          : Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ]),
        "Random Forest"  : RandomForestRegressor(
            n_estimators=200, max_depth=6,
            min_samples_leaf=3, random_state=RANDOM_STATE,
        ),
        "Gradient Boost" : GradientBoostingRegressor(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, random_state=RANDOM_STATE,
        ),
    }

    tahun_list = sorted(df[df["Tahun"] >= TAHUN_TEST_START]["Tahun"].unique())
    cv_results = {}

    print(f"{'Model':<20} {'RMSE (poin%)':>13} {'MAE (poin%)':>13} {'R²':>8}")
    print("-" * 58)

    for name, model in models.items():
        rmses, maes, r2s = [], [], []
        for t in tahun_list:
            train = df[df["Tahun"] <  t]
            test  = df[df["Tahun"] == t]
            if len(test) == 0:
                continue

            model.fit(train[FEATURES], np.log1p(train[TARGET]))
            y_pred = np.expm1(model.predict(test[FEATURES]))
            y_true = test[TARGET].values

            rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))
            maes.append(mean_absolute_error(y_true, y_pred))
            r2s.append(r2_score(y_true, y_pred))

        cv_results[name] = {
            "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
            "mae_mean" : np.mean(maes),  "mae_std" : np.std(maes),
            "r2_mean"  : np.mean(r2s),   "r2_std"  : np.std(r2s),
            "model"    : model,
        }
        print(
            f"{name:<20} "
            f"{np.mean(rmses):>9.4f}±{np.std(rmses):.3f} "
            f"{np.mean(maes):>9.4f}±{np.std(maes):.3f} "
            f"{np.mean(r2s):>8.4f}"
        )

    best_name = max(cv_results, key=lambda k: cv_results[k]["r2_mean"])
    print(f"\n[OK] Model terbaik: {best_name} "
          f"(R²={cv_results[best_name]['r2_mean']:.4f})")
    return cv_results, best_name, tahun_list


# ── 4. Per-tahun Detail & Final Fit ──────────────────────────────────────────
def final_fit(df: pd.DataFrame, cv_results: dict,
              best_name: str, tahun_list: list):
    print(f"\n── 4. DETAIL PER-TAHUN ({best_name}) ────────────────")
    print(f"{'Tahun':>8} {'N':>4} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-" * 42)

    best_model = cv_results[best_name]["model"]
    all_y_true, all_y_pred = [], []

    for t in tahun_list:
        train = df[df["Tahun"] <  t]
        test  = df[df["Tahun"] == t]
        if len(test) == 0:
            continue

        best_model.fit(train[FEATURES], np.log1p(train[TARGET]))
        y_pred = np.expm1(best_model.predict(test[FEATURES]))
        y_true = test[TARGET].values

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        print(
            f"{t:>8} {len(test):>4} "
            f"{np.sqrt(mean_squared_error(y_true, y_pred)):>8.4f} "
            f"{mean_absolute_error(y_true, y_pred):>8.4f} "
            f"{r2_score(y_true, y_pred):>8.4f}"
        )

    # Final fit pakai seluruh data untuk simpan model
    best_model.fit(df[FEATURES], np.log1p(df[TARGET]))

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        print(f"\n[OK] Feature Importance:")
        imp = pd.Series(best_model.feature_importances_, index=FEATURES)
        for feat, val in imp.sort_values(ascending=False).items():
            bar = "█" * int(val * 40)
            print(f"     {feat:<35} {val:.4f}  {bar}")

    return best_model, np.array(all_y_true), np.array(all_y_pred)


# ── 5. Simpan Model ───────────────────────────────────────────────────────────
def save_model(model, path: str):
    print(f"\n── 5. SIMPAN MODEL ──────────────────────────────────")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[OK] Model tersimpan → {path}")


# ── 6. Visualisasi ────────────────────────────────────────────────────────────
def plot_results(y_true, y_pred, cv_results: dict, tahun_list: list, df: pd.DataFrame):
    print(f"\n── 6. VISUALISASI ───────────────────────────────────")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Evaluasi Model Regresi — Prediksi Kemiskinan\n"
                 "(Walk-forward validation, skala asli poin%)", fontsize=13)

    # Plot 1: Actual vs Predicted
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.55, color="#378ADD",
               edgecolors="white", linewidth=0.4, s=40)
    lims = [min(y_true.min(), y_pred.min()) - 1,
            max(y_true.max(), y_pred.max()) + 1]
    ax.plot(lims, lims, "r--", linewidth=1, label="Prediksi sempurna")
    ax.set_xlabel("Aktual (%)")
    ax.set_ylabel("Prediksi (%)")
    ax.set_title("Aktual vs Prediksi")
    ax.legend(fontsize=9)

    # Plot 2: Residual
    ax = axes[1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.55, color="#1D9E75",
               edgecolors="white", linewidth=0.4, s=40)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediksi (%)")
    ax.set_ylabel("Residual (aktual − prediksi)")
    ax.set_title("Residual Plot")

    # Plot 3: R² per tahun (Random Forest)
    ax = axes[2]
    rf_model = cv_results["Random Forest"]["model"]
    r2_per_tahun = []
    for t in tahun_list:
        train = df[df["Tahun"] <  t]
        test  = df[df["Tahun"] == t]
        if len(test) == 0:
            continue
        rf_model.fit(train[FEATURES], np.log1p(train[TARGET]))
        yp = np.expm1(rf_model.predict(test[FEATURES]))
        r2_per_tahun.append(r2_score(test[TARGET].values, yp))

    ax.bar([str(t) for t in tahun_list], r2_per_tahun,
           color="#378ADD", alpha=0.85)
    ax.axhline(np.mean(r2_per_tahun), color="red", linestyle="--",
               linewidth=1, label=f"Rata-rata R²={np.mean(r2_per_tahun):.3f}")
    ax.set_xlabel("Tahun test")
    ax.set_ylabel("R²")
    ax.set_title("R² per Tahun (Random Forest)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out_path = os.path.join(MODEL_DIR, "evaluasi_regresi.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Plot tersimpan → {out_path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Model Regresi — Prediksi Kemiskinan ===\n")

    df       = load_and_clean(CLEAN_DATA_PATH)
    df_feat  = build_features(df)

    cv_results, best_name, tahun_list = walk_forward_validate(df_feat)
    best_model, y_true, y_pred        = final_fit(df_feat, cv_results, best_name, tahun_list)

    save_model(best_model, MODEL_PATH)
    plot_results(y_true, y_pred, cv_results, tahun_list, df_feat)

    print("\n=== Selesai ✅ ===")