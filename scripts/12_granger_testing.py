"""
12_granger_causality.py

Per-stock Granger causality tests of sentiment variables on daily returns (r0),
using the frozen COMMON_LAG from VAR lag selection. Training window only.

Five tests per stock:
  - Four bivariate tests (one sentiment variable at a time)
  - One joint block test (all four sentiment variables together)
Each compares a restricted model (own return lags only) against an
unrestricted model (own lags + sentiment lags) via a nested F-test.
"""
from runtime_log import timed
import numpy as np
import pandas as pd
from scipy import stats

from config import (MODEL_DATASET, GRANGER_RESULTS, COMMON_LAG,
                    TRAIN_START_DATE, TRAIN_END_DATE)

TARGET = 'r0'
SENTIMENT_VARS = ['sentiment_score_z', 'sentiment_volume_z',
                  'universe_sentiment_score', 'universe_sentiment_volume']

MIN_ROWS = 250          # same guard as VAR lag selection
STD_FLOOR = 1e-8        # near-constant column guard
ALPHA_BASELINE = 0.05   # nominal size for the eyeball-vs-baseline rule

# Block 2 - Load and filter to the training window

LOAD_COLS = ['ISIN', 'bb_tcm', 'date', TARGET] + SENTIMENT_VARS

df = pd.read_parquet(MODEL_DATASET, columns=LOAD_COLS)
df = df[(df['date'] >= TRAIN_START_DATE) & (df['date'] <= TRAIN_END_DATE)]

print(f"Training panel: {len(df):,} rows, "
      f"{df['ISIN'].nunique():,} ISINs, "
      f"{df['date'].min().date()} to {df['date'].max().date()}")

# Block 3 - Lag-matrix helper

OWN_LAG_COLS = [f'{TARGET}_l{i}' for i in range(1, COMMON_LAG + 1)]

def sent_lag_cols(var):
    """Lagged column names for one sentiment variable."""
    return [f'{var}_l{i}' for i in range(1, COMMON_LAG + 1)]

def build_lag_frame(g):
    """
    From one stock's chronologically sorted data, build a frame containing
    y (r0 at time t) and every lagged regressor at t-1 ... t-COMMON_LAG.
    Rows with NaNs from the shift (the first COMMON_LAG rows) are dropped.
    """
    g = g.sort_values('date')
    out = {'y': g[TARGET]}
    for lag in range(1, COMMON_LAG + 1):
        out[f'{TARGET}_l{lag}'] = g[TARGET].shift(lag)
        for var in SENTIMENT_VARS:
            out[f'{var}_l{lag}'] = g[var].shift(lag)
    return pd.DataFrame(out).dropna()

# Block 4 - OLS and F-test helpers

def ols_rss(y, X):
    """Fit OLS by least squares and return the residual sum of squares."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return resid @ resid

def nested_f_test(y, X_restricted, X_unrestricted):
    """
    Standard nested-model F-test.
    H0: the extra coefficients in the unrestricted model are all zero.
    Returns (F, p, n).
    """
    n = len(y)
    k_u = X_unrestricted.shape[1]              # params in unrestricted model
    q = k_u - X_restricted.shape[1]            # number of restrictions
    rss_r = ols_rss(y, X_restricted)
    rss_u = ols_rss(y, X_unrestricted)
    f_stat = ((rss_r - rss_u) / q) / (rss_u / (n - k_u))
    p_val = stats.f.sf(f_stat, q, n - k_u)
    return f_stat, p_val, n

# Block 5 - Per-stock loop

results = []
skipped = []
with timed('granger', 'selection'):
    for isin, g in df.groupby('ISIN', sort=False):
        ticker = g['bb_tcm'].iloc[0]
        try:
            lagged = build_lag_frame(g)

            if len(lagged) < MIN_ROWS:
                skipped.append({'ISIN': isin, 'bb_tcm': ticker,
                                'reason': f'insufficient rows ({len(lagged)})'})
                continue

            y = lagged['y'].to_numpy()
            X_r = np.column_stack([np.ones(len(lagged)),
                                   lagged[OWN_LAG_COLS].to_numpy()])

            # Near-constant guard: a variable with (almost) no variation for this
            # stock makes the design matrix rank-deficient, so its test is skipped.
            usable = [v for v in SENTIMENT_VARS
                      if lagged[sent_lag_cols(v)].std().min() > STD_FLOOR]

            # --- Tests 1-4: bivariate, one sentiment variable at a time ---
            for var in SENTIMENT_VARS:
                if var not in usable:
                    skipped.append({'ISIN': isin, 'bb_tcm': ticker,
                                    'reason': f'near-constant: {var}'})
                    continue
                X_u = np.column_stack(
                    [X_r, lagged[sent_lag_cols(var)].to_numpy()])
                f_stat, p_val, n = nested_f_test(y, X_r, X_u)
                results.append({'ISIN': isin, 'bb_tcm': ticker, 'test': var,
                                'f_stat': f_stat, 'p_value': p_val, 'n_obs': n})

            # --- Test 5: joint block test, all four together ---
            if len(usable) == len(SENTIMENT_VARS):
                all_sent_cols = [c for v in SENTIMENT_VARS
                                 for c in sent_lag_cols(v)]
                X_u = np.column_stack([X_r, lagged[all_sent_cols].to_numpy()])
                f_stat, p_val, n = nested_f_test(y, X_r, X_u)
                results.append({'ISIN': isin, 'bb_tcm': ticker,
                                'test': 'joint_all_sentiment',
                                'f_stat': f_stat, 'p_value': p_val, 'n_obs': n})
            else:
                skipped.append({'ISIN': isin, 'bb_tcm': ticker,
                                'reason': 'joint test skipped (degenerate variable)'})

        except Exception as e:
            skipped.append({'ISIN': isin, 'bb_tcm': ticker,
                            'reason': f'error: {e}'})

print(f"Tests completed: {len(results):,} | Skips/failures logged: {len(skipped):,}")

# Block 6 - Store results

res_df = pd.DataFrame(results)
res_df.to_parquet(GRANGER_RESULTS, index=False)
print(f"Saved {len(res_df):,} test results to {GRANGER_RESULTS}")

if skipped:
    skip_df = pd.DataFrame(skipped)
    print("\nSkip summary:")
    print(skip_df['reason'].str.split(':').str[0].value_counts())

# Block 7 - Aggregate summary against the 5% baseline

def rejection_rate(p_series, alpha):
    return (p_series < alpha).mean()

summary = (res_df.groupby('test')['p_value']
           .agg(n_stocks='count',
                median_p='median',
                rej_1pct=lambda p: rejection_rate(p, 0.01),
                rej_5pct=lambda p: rejection_rate(p, 0.05),
                rej_10pct=lambda p: rejection_rate(p, 0.10))
           .sort_values('rej_5pct', ascending=False))

print("\n=== Granger causality: rejection rates across the universe ===")
print("(Chance baseline under H0: ~1%, ~5%, ~10% respectively)\n")
print(summary.round(4).to_string())