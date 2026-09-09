"""13_lasso.py — Pooled LASSO feature selection, two control-set runs.

Run A ('granger_mirror'): r0 + r_1 controls  -> direct comparison with Granger.
Run B ('full_controls'):  all 7 pricing controls -> selection under the
                          evaluation environment (Gu et al. framing).
Sentiment candidates identical in both runs: the 4 Granger-tested variables
at 2 lags (unshifted = Granger lag 1; shift(1) = Granger lag 2).
"""
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, lasso_path
from sklearn.model_selection import TimeSeriesSplit

from config import MODEL_DATASET, LASSO_RESULTS, TRAIN_START_DATE, TRAIN_END_DATE

TARGET = 'fwd_r1'

SENTIMENT_VARS = ['sentiment_score_z', 'sentiment_volume_z',
                  'universe_sentiment_score', 'universe_sentiment_volume']

PRICING_RUN_A = ['r0', 'r_1']                                   # Granger's 2 return lags
PRICING_RUN_B = ['r0', 'r_1', 'r_5_2', 'r_10_6',
                 'pvma', 'r0_z20', 'Sector_z20']

N_SPLITS = 5          # TimeSeriesSplit folds inside LassoCV
N_ALPHAS = 100        # points on the regularisation path
MAX_ITER = 10_000     # coordinate-descent iteration cap
COEF_TOL = 1e-10      # up with the other constants: below this = numerically zero




t0 = time.perf_counter()

df = pd.read_parquet(MODEL_DATASET)
df['date'] = pd.to_datetime(df['date'])

# Lag-2 sentiment: yesterday's value, per stock
df = df.sort_values(['ISIN', 'date'])
for var in SENTIMENT_VARS:
    df[f'{var}_lag2'] = df.groupby('ISIN')[var].shift(1)

SENT_ALL = SENTIMENT_VARS + [f'{v}_lag2' for v in SENTIMENT_VARS]

# Training window only — MUST match the mask used in 11/12
train = df[(df['date'] >= TRAIN_START_DATE) & (df['date'] <= TRAIN_END_DATE)].copy()

# Chronological order so TimeSeriesSplit folds are temporal blocks
train = train.sort_values(['date', 'ISIN']).reset_index(drop=True)

t_prep = time.perf_counter() - t0


runs = {'Run A (Granger comparison)': PRICING_RUN_A + SENT_ALL,
        'Run B (All pricing features':  PRICING_RUN_B + SENT_ALL}

results = []

for run_name, features in runs.items():
    t_run = time.perf_counter()

    data = train.dropna(subset=features + [TARGET])
    X = data[features].to_numpy()
    y = data[TARGET].to_numpy()

    X = StandardScaler().fit_transform(X)

    model = LassoCV(
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        n_alphas=N_ALPHAS,
        max_iter=MAX_ITER,
        n_jobs=-1,
    )

    model.fit(X, y)

    # --- Regularisation path: entry-order ranking ---
    alphas_path, coefs_path, _ = lasso_path(X, y, alphas=model.alphas_)
    # coefs_path shape: (n_features, n_alphas), alphas descending

    entry_alpha = []
    for i, feat in enumerate(features):
        nonzero = np.flatnonzero(np.abs(coefs_path[i]) > COEF_TOL)  # <-- here
        entry_alpha.append(alphas_path[nonzero[0]] if len(nonzero) else np.nan)

    path_rank = (pd.Series(entry_alpha, index=features)
                 .rank(ascending=False))  # rank 1 = strongest

    runtime = time.perf_counter() - t_run

    for j, (feat, coef) in enumerate(zip(features, model.coef_)):
        results.append({
            'run': run_name,
            'feature': feat,
            'coefficient': coef,
            'selected': np.abs(coef) > COEF_TOL,
            'entry_alpha': entry_alpha[j],
            'path_rank': path_rank[feat],
            'alpha': model.alpha_,
            'n_rows': len(data),
            'n_features': len(features),
            'runtime_sec': runtime,
            'prep_runtime_sec': t_prep,
        })

    kept = (model.coef_ != 0).sum()
    print(f'{run_name}: alpha={model.alpha_:.6g}, '
          f'{kept}/{len(features)} features kept, '
          f'{len(data):,} rows, {runtime:.1f}s')

res = pd.DataFrame(results)
res.to_parquet(LASSO_RESULTS, index=False)
print(f'Saved {len(res)} rows -> {LASSO_RESULTS}')