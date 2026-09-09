"""

Selects a single common VAR lag order for the Granger causality tests.

Method:
    - For each stock (ISIN), fit VARs of order 1..MAXLAGS on the TRAINING
      window only, using r0 plus the four sentiment features.
    - Record the lag selected by BIC (plus AIC and HQIC for robustness).
    - Aggregate across stocks: report both the modal and median BIC lag.


"""

import warnings
from runtime_log import timed
import numpy as np
import pandas as pd
import config
import importlib
importlib.reload(config)

from statsmodels.tsa.api import VAR

from config import (
    MODEL_DATASET,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    VAR_LAG_RESULTS,
)


# Constants

# Variables entering each stock's VAR. Defined once
VAR_COLUMNS = [
    "r0",
    "sentiment_score_z",
    "sentiment_volume_z",
    "universe_sentiment_score",
    "universe_sentiment_volume",
]

MAXLAGS = 10        # candidate lag orders 1..10 (two trading weeks)
MIN_OBS = 250       # ~1 trading year; guards the obs-to-parameters ratio
STD_FLOOR = 1e-8    # columns with std below this are treated as constant

with timed("granger", "VAR_lag_selection"):
# Load and window the data
    df = pd.read_parquet(MODEL_DATASET)

    # Training window only
    mask = (df["date"] >= TRAIN_START_DATE) & (df["date"] <= TRAIN_END_DATE)
    train = df.loc[mask, ["ISIN","bb_tcm", "date", "Sector"] + VAR_COLUMNS].copy()

    print(f"Training window: {train['date'].min().date()} to {train['date'].max().date()}")
    print(f"Rows: {len(train):,} | Stocks: {train['ISIN'].nunique():,}")

    # loop per-stock
    records = []

    for isin, g in train.groupby("ISIN", sort=False):

        bb_tcm = g["bb_tcm"].iloc[0]
        sector = g["Sector"].iloc[0]

        # Date-indexed frame of just the VAR columns, in time order.
        stock_df = (
            g.sort_values("date")
             .set_index("date")[VAR_COLUMNS]
             .dropna()
        )

        record = {
            "ISIN": isin,
            "bb_tcm": bb_tcm,
            "Sector": sector,
            "n_obs": len(stock_df),
            "bic_lag": np.nan,
            "aic_lag": np.nan,
            "hqic_lag": np.nan,
            "status": "ok",
            "error_msg": "",
        }

        # Guard 1: enough observations for a stable fit at MAXLAGS.
        if len(stock_df) < MIN_OBS:
            record["status"] = "too_few_obs"
            records.append(record)
            continue

        # Guard 2: near-constant columns make the design matrix
        # ill-conditioned (typically stocks with almost no news coverage).
        stds = stock_df.std()
        if (stds < STD_FLOOR).any():
            record["status"] = "degenerate_column"
            record["error_msg"] = ",".join(stds.index[stds < STD_FLOOR])
            records.append(record)
            continue

        # Fit and select. select_order fits VAR(1)..VAR(MAXLAGS) and returns
        # the order minimising each information criterion.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sel = VAR(stock_df).select_order(maxlags=MAXLAGS)
            record["bic_lag"] = sel.selected_orders["bic"]
            record["aic_lag"] = sel.selected_orders["aic"]
            record["hqic_lag"] = sel.selected_orders["hqic"]
        except Exception as exc:
            record["status"] = "fit_failed"
            record["error_msg"] = f"{type(exc).__name__}: {exc}"

        records.append(record)

    results = pd.DataFrame(records)

# Save -- one file tells the whole story, including exclusions

results.to_parquet(VAR_LAG_RESULTS, index=False)
print(f"\nSaved per-stock results to: {VAR_LAG_RESULTS}")

# Aggregate and report

print("\n--- Status summary ---")
print(results["status"].value_counts().to_string())

ok = results.loc[results["status"] == "ok"]

if ok.empty:
    raise RuntimeError("No successful VAR fits -- inspect the results file.")

print(f"\nSuccessful fits: {len(ok):,} of {len(results):,} stocks")

print("\n--- BIC-selected lag distribution ---")
dist = ok["bic_lag"].value_counts().sort_index()
for lag, count in dist.items():
    pct = 100 * count / len(ok)
    print(f"  lag {int(lag):>2}: {count:>5,} stocks ({pct:5.1f}%)")

modal_lag = int(ok["bic_lag"].mode().iloc[0])
median_lag = float(ok["bic_lag"].median())

print(f"\nModal BIC lag:  {modal_lag}")
print(f"Median BIC lag: {median_lag}")

# Robustness: the same aggregates under the other criteria.
print("\n--- Robustness (modal | median) ---")
for crit in ["aic_lag", "hqic_lag"]:
    print(f"  {crit[:-4].upper():>4}: "
          f"{int(ok[crit].mode().iloc[0])} | {ok[crit].median()}")

print(
    "\nNext step: freeze the chosen common lag as COMMON_LAG in config.py "
    "(a hard-coded literal, per the project convention), then import it "
    "in the Granger script."
)