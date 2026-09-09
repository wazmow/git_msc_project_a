"""
14_rf_testing.py

Random Forest feature importance across four model specifications:
  A_baseline    : pricing features only
  B_all         : pricing + all four sentiment variables (lags 1-2)
  C_granger     : pricing + Granger-selected sentiment variables (lags 1-2)
  D_lasso_path  : pricing + LASSO path-ranking substitute (lags 1-2)

Importance measured two ways per model:
  - impurity-based (feature_importances_)
  - permutation importance on the test window

Outputs:
  RF_RESULTS      : long-format importances per model/feature
  RF_PERFORMANCE  : one row per model with OOS metrics + runtimes
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
from pathlib import Path

from config import (
    MODEL_DATASET,          # input parquet path
    RF_RESULTS,             # output parquet path (importances)
    RF_PERFORMANCE,         # output path (metrics + runtimes)
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
)
from runtime_log import timed

#Define hard variables
TARGET = "fwd_r1"
PRICING_FEATURES = ["r0", "r_1", "r_5_2", "r_10_6", "pvma", "r0_z20", "Sector_z20"]
SENTIMENT_VARS = ["sentiment_score_z","sentiment_volume_z","universe_sentiment_score","universe_sentiment_volume"]
N_LAGS = 2  # matches COMMON_LAG frozen in config; lag 1 = unshifted, lag 2 = shift(1)
SEED = 42

RF_PARAMS = dict(
    n_estimators=300,
    min_samples_leaf=50,
    max_features="sqrt",
    random_state=SEED,
    n_jobs=-1,
)

PERM_REPEATS = 5  # permutation importance repeats (test set)
STD_FLOOR = 1e-8  # near-constant column check


def make_lag_cols(vars_, n_lags=N_LAGS):
    """Expand variable names into their lagged column names."""
    return [f"{v}_lag{k}" for v in vars_ for k in range(1, n_lags + 1)]


# Model specifications: feature lists built from frozen constants above.
MODELS = {
    "A_baseline":   PRICING_FEATURES,
    "B_all":        PRICING_FEATURES + make_lag_cols(SENTIMENT_VARS),
    "C_granger":    PRICING_FEATURES + make_lag_cols(
                        ["universe_sentiment_volume",
                         "sentiment_volume_z",
                         "sentiment_score_z"]),
    "D_lasso_path": PRICING_FEATURES + make_lag_cols(
                        ["universe_sentiment_volume"]),
}

# Section 2: Load data and construct lags

def load_and_prepare(target=TARGET):
    """Load MODEL_DATASET, build sentiment lag columns, apply guards.

    Returns the prepared DataFrame plus a small audit dict.
    """
    with timed("rf", "load_dataset"):
        df = pd.read_parquet(MODEL_DATASET)

    # Defensive: sort so per-ISIN shifts are chronologically correct.
    df = df.sort_values(["ISIN", "date"]).reset_index(drop=True)

    # Build lag columns per ISIN
    # With the fwd_r1 target design, the unshifted sentiment column already
    # occupies the lag-1 position; shift(1) within each ISIN gives lag 2.
    with timed("rf", "build_lags"):
        grouped = df.groupby("ISIN", sort=False)
        for var in SENTIMENT_VARS:
            df[f"{var}_lag1"] = df[var]
            df[f"{var}_lag2"] = grouped[var].shift(1)

    # Assemble the full column set needed by any model
    all_features = sorted(set().union(*MODELS.values()))
    keep_cols = ["ISIN", "date", target] + all_features
    df = df[keep_cols]

    # Drop rows unusable by the widest model
    # lag2 columns are NaN on each ISIN's first row; fwd_r1 is NaN on each
    # ISIN's final row. One common sample across all four models means every
    # importance comparison is on identical rows.
    n_before = len(df)
    df = df.dropna(subset=all_features + [target]).reset_index(drop=True)
    n_dropped = n_before - len(df)

    # Guard: near-constant columns
    stds = df[all_features].std()
    near_constant = stds[stds < STD_FLOOR].index.tolist()
    if near_constant:
        raise ValueError(
            f"Near-constant feature columns (std < {STD_FLOOR}): {near_constant}"
        )

    audit = {
        "rows_loaded": n_before,
        "rows_dropped_nan": n_dropped,
        "rows_final": len(df),
        "n_isins": df["ISIN"].nunique(),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
    }
    return df, audit

def train_test_split_dates(df):
    """Split on the frozen date literals from config.py."""
    train_mask = (df["date"] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)
    test_mask = (df["date"] >= TEST_START_DATE) & (df["date"] <= TEST_END_DATE)

    train = df.loc[train_mask]
    test = df.loc[test_mask]

    # Guards: no overlap, no empty windows, no leakage.
    if len(train) == 0 or len(test) == 0:
        raise ValueError("Empty train or test window - check config dates.")
    if train["date"].max() >= test["date"].min():
        raise ValueError("Train window overlaps test window - check config dates.")

    return train, test

# Section 3: Fit loop - one RF per model specification

def run_model(model_name, features, train, test,target=TARGET):
    """Fit one RF specification, compute both importance measures and
    OOS metrics. Returns (importance_df, performance_row)."""

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    # Fit
    t0 = time.perf_counter()
    with timed("rf",f"fit_{model_name}"):
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_train, y_train)
    fit_secs = time.perf_counter() - t0

    # Importance measure 1: impurity (train-side, effectively free)
    imp_impurity = pd.Series(rf.feature_importances_, index=features)

    # Importance measure 2: permutation (test-side, expensive)
    t0 = time.perf_counter()
    with timed("rf",f"perm_{model_name}"):
        perm = permutation_importance(
            rf, X_test, y_test,
            n_repeats=PERM_REPEATS,
            random_state=SEED,
            n_jobs=-1,
            scoring="r2",
        )
    perm_secs = time.perf_counter() - t0

    imp_perm = pd.Series(perm.importances_mean, index=features)
    imp_perm_std = pd.Series(perm.importances_std, index=features)

    # OOS performance
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = mean_absolute_error(y_test, y_pred)

    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_test)))
    train_mean = float(y_train.mean())
    bench_mae = float(np.mean(np.abs(y_test - train_mean)))
    bench_dir = float(np.mean(np.sign(train_mean) == np.sign(y_test)))

    r2_ct = float(1 - np.sum((y_test - y_pred) ** 2)
                    / np.sum((y_test - train_mean) ** 2))

    # Assemble outputs
    importance_df = pd.DataFrame({
        "model": model_name,
        "feature": features,
        "importance_impurity": imp_impurity.values,
        "importance_permutation": imp_perm.values,
        "perm_std": imp_perm_std.values,
    })

    performance_row = {
        "model": model_name,
        "n_features": len(features),
        "r2_oos": r2,
        "r2_oos_ct": r2_ct,
        "rmse_oos": rmse,
        "mae_oos": mae,
        "fit_secs": fit_secs,
        "perm_secs": perm_secs,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "dir_acc_oos": dir_acc,
        "bench_mae_oos": bench_mae,
        "bench_dir_acc": bench_dir,
    }

    return importance_df, performance_row

# Section 4: Main - run all models, save results, print summary

def tagged_path(base_path, tag):
    """Insert a tag before the file extension: rf_results_1.parquet ->
    rf_results_1_fwd_r5.parquet."""
    p = Path(base_path)
    return p.with_name(f"{p.stem}_{tag}{p.suffix}")


def run_all(target=TARGET, save=True):
    """Run all four model specifications against one target variable."""
    with timed("rf", f"total_{target}"):
        df, audit = load_and_prepare(target=target)
        train, test = train_test_split_dates(df)

        importance_frames = []
        performance_rows = []

        for model_name, features in MODELS.items():
            print(f"\n--- Running {model_name} "
                  f"({len(features)} features, target={target}) ---")
            imp_df, perf_row = run_model(
                model_name, features, train, test, target=target
            )
            imp_df["target"] = target
            perf_row["target"] = target
            importance_frames.append(imp_df)
            performance_rows.append(perf_row)
            print(f"    OOS R2: {perf_row['r2_oos']:.6f} | "
                  f"fit {perf_row['fit_secs']:.1f}s | "
                  f"perm {perf_row['perm_secs']:.1f}s")

        rf_results = pd.concat(importance_frames, ignore_index=True)
        rf_performance = pd.DataFrame(performance_rows)

        if save:
            rf_results.to_parquet(tagged_path(RF_RESULTS, target), index=False)
            rf_performance.to_csv(tagged_path(RF_PERFORMANCE, target), index=False)

    return rf_results, rf_performance, audit


def main():
    with timed("rf","total"):
        df, audit = load_and_prepare()
        train, test = train_test_split_dates(df)

        importance_frames = []
        performance_rows = []

        for model_name, features in MODELS.items():
            print(f"\n--- Running {model_name} "
                  f"({len(features)} features) ---")
            imp_df, perf_row = run_model(model_name, features, train, test)
            importance_frames.append(imp_df)
            performance_rows.append(perf_row)
            print(f"    OOS R2: {perf_row['r2_oos']:.6f} | "
                  f"fit {perf_row['fit_secs']:.1f}s | "
                  f"perm {perf_row['perm_secs']:.1f}s")

        # Save outputs
        rf_results = pd.concat(importance_frames, ignore_index=True)
        rf_performance = pd.DataFrame(performance_rows)

        rf_results.to_parquet(RF_RESULTS, index=False)
        rf_performance.to_csv(RF_PERFORMANCE, index=False)

    # Run summary
    print("\n" + "=" * 70)
    print("RUN SUMMARY - 14_rf_testing.py")
    print("=" * 70)
    for k, v in audit.items():
        print(f"  {k:>20}: {v}")
    print(f"  {'train window':>20}: {TRAIN_START_DATE} to {TRAIN_END_DATE} "
          f"({len(train):,} rows)")
    print(f"  {'test window':>20}: {TEST_START_DATE} to {TEST_END_DATE} "
          f"({len(test):,} rows)")

    print("\nPerformance comparison:")
    cols = ["model", "n_features", "r2_oos", "rmse_oos",
            "fit_secs", "perm_secs"]
    print(rf_performance[cols].to_string(index=False))

    print("\nTop 5 features per model (permutation importance):")
    for model_name in MODELS:
        top = (rf_results[rf_results["model"] == model_name]
               .nlargest(5, "importance_permutation")
               [["feature", "importance_permutation", "perm_std"]])
        print(f"\n  {model_name}:")
        print(top.to_string(index=False))

    print(f"\nSaved: {RF_RESULTS}")
    print(f"Saved: {RF_PERFORMANCE}")


if __name__ == "__main__":
    main()