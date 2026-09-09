import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller


MIN_OBS = 100          # minimum non-missing observations per (ISIN, variable)
ALPHA = 0.05           # significance level for rejection-rate summary
PROGRESS_EVERY = 100   # print progress every N ISINs

def adf_test(series: pd.Series, isin: str, var: str) -> dict:
    """Run ADF on a cleaned series; return a tidy result dict."""
    series = series.dropna()

    if len(series) < MIN_OBS:
        return {'ISIN': isin, 'var': var, 'status': 'too_short',
                'nobs': len(series), 'adf_stat': np.nan, 'pvalue': np.nan,
                'usedlag': np.nan}

    if series.nunique() <= 1:
        return {'ISIN': isin, 'var': var, 'status': 'constant',
                'nobs': len(series), 'adf_stat': np.nan, 'pvalue': np.nan,
                'usedlag': np.nan}

    try:
        adf_stat, pvalue, usedlag, nobs, _, _ = adfuller(
            series, regression='c', autolag='AIC'
        )
        return {'ISIN': isin, 'var': var, 'status': 'ok',
                'nobs': nobs, 'adf_stat': adf_stat, 'pvalue': pvalue,
                'usedlag': usedlag}
    except Exception as e:
        return {'ISIN': isin, 'var': var, 'status': f'error: {e}',
                'nobs': len(series), 'adf_stat': np.nan, 'pvalue': np.nan,
                'usedlag': np.nan}


def adf_run(df: pd.DataFrame, stock_vars:list,universe_vars:list,ADF_RESULTS:str) -> pd.DataFrame:
    print(f"Loaded MODEL_DATASET: {df.shape[0]:,} rows, "
          f"{df['ISIN'].nunique():,} unique ISINs")

    # Per-stock tests
    # ---------------------------------------------------------------------------

    adf_results = []
    n_isins = df['ISIN'].nunique()

    for i, (isin, g) in enumerate(df.groupby('ISIN'), start=1):
        g = g.sort_values('date')
        for var in stock_vars:
            adf_results.append(adf_test(g[var], isin, var))

        if i % PROGRESS_EVERY == 0 or i == n_isins:
            print(f"  processed {i:,} / {n_isins:,} ISINs")

    # ---------------------------------------------------------------------------
    # Universe-level tests (one series per variable, deduplicated by date)
    # ---------------------------------------------------------------------------

    universe = (df[['date'] + universe_vars]
                .drop_duplicates(subset='date')
                .sort_values('date'))

    for var in universe_vars:
        adf_results.append(adf_test(universe[var], 'UNIVERSE', var))

    # ---------------------------------------------------------------------------
    # Collate
    # ---------------------------------------------------------------------------

    results = pd.DataFrame(adf_results)
    results.to_parquet(ADF_RESULTS)
    print(f"\nSaved per-series results: {results.shape[0]:,} rows "
          f"-> {ADF_RESULTS}")

    # ---------------------------------------------------------------------------
    # Summary: rejection rates + Maddala-Wu combined test (per-stock vars)
    # ---------------------------------------------------------------------------

    ok = results[results['status'] == 'ok']

    summary_rows = []
    for var in stock_vars:
        sub = ok[ok['var'] == var]
        n_tested = len(sub)
        n_skipped = len(results[(results['var'] == var) &
                                (results['status'] != 'ok') &
                                (results['ISIN'] != 'UNIVERSE')])
        if n_tested == 0:
            summary_rows.append({'var': var, 'n_tested': 0,
                                 'n_skipped': n_skipped})
            continue

        reject_rate = (sub['pvalue'] < ALPHA).mean()

        # Maddala-Wu (Fisher) combination: -2 * sum(ln p_i) ~ chi2(2N)
        pvals = sub['pvalue'].clip(lower=1e-300)  # guard against log(0)
        mw_stat = -2.0 * np.log(pvals).sum()
        mw_pvalue = stats.chi2.sf(mw_stat, df=2 * n_tested)

        summary_rows.append({
            'var': var,
            'n_tested': n_tested,
            'n_skipped': n_skipped,
            'reject_rate_5pct': reject_rate,
            'median_pvalue': sub['pvalue'].median(),
            'mw_stat': mw_stat,
            'mw_pvalue': mw_pvalue,
        })

    summary = pd.DataFrame(summary_rows)

    # Universe variables: single-series results reported directly
    uni = results[results['ISIN'] == 'UNIVERSE'][
        ['var', 'status', 'nobs', 'adf_stat', 'pvalue', 'usedlag']]

    # ---------------------------------------------------------------------------
    # Output for results log
    # ---------------------------------------------------------------------------

    pd.set_option('display.float_format', lambda x: f'{x:.4g}')

    print("\n=== ADF summary: per-stock variables ===")
    print(summary.to_string(index=False))

    print("\n=== ADF results: universe variables ===")
    print(uni.to_string(index=False))

    print("\n=== Skipped series by reason ===")
    print(results[results['status'] != 'ok']
          .groupby(['var', 'status']).size().rename('count').reset_index()
          .to_string(index=False))

    return results