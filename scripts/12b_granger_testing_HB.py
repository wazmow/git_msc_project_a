"""
12b_granger_bh_correction.py

Benjamini-Hochberg FDR correction applied to the per-stock Granger p-values,
separately within each test. Settles the borderline selection cases by
reporting how many rejections survive multiple-testing correction.
"""

import pandas as pd
from statsmodels.stats.multitest import multipletests

from config import GRANGER_RESULTS

FDR_ALPHA = 0.05   # target false discovery rate

res_df = pd.read_parquet(GRANGER_RESULTS)

rows = []
for test, g in res_df.groupby('test'):
    reject, p_adj, *_ = multipletests(g['p_value'], alpha=FDR_ALPHA,
                                      method='fdr_bh')
    rows.append({
        'test': test,
        'n_stocks': len(g),
        'raw_rej_5pct': (g['p_value'] < 0.05).mean(),
        'bh_discovery_rate': reject.mean(),
        'n_bh_discoveries': int(reject.sum()),
        'min_p_adj': p_adj.min(),
    })

bh_summary = (pd.DataFrame(rows)
              .set_index('test')
              .sort_values('bh_discovery_rate', ascending=False))

print(f"=== BH-corrected Granger results (FDR alpha = {FDR_ALPHA}) ===\n")
print(bh_summary.round(4).to_string())
bh_summary.to_clipboard()