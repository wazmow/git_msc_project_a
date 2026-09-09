import pandas as pd
import numpy as np
### THIS WOULDN'T  WORK YET MATT. YOU NEED TO REMOVE THE FIELDS AND MAKE IT GENERIC
def add_loo_zscore(df, value_col, z_col, min_group_size):
    """
    Adds a leave-one-out z-score column: how unusual is this stock's value
    relative to the OTHER covered stocks in the same (Sector, trading_day)
    group. Groups smaller than min_group_size get z = 0.
    """
    grp = df.groupby(['Sector', 'trading_day'], sort=False)

    x = df[value_col]
    n = grp[value_col].transform('count')
    s = grp[value_col].transform('sum')

    df['_sq'] = x ** 2
    q = df.groupby(['Sector', 'trading_day'], sort=False)['_sq'].transform('sum')
    df.drop(columns='_sq', inplace=True)

    # Mean of the group excluding the stock itself
    loo_mean = (s - x) / (n - 1)

    # Sample variance (ddof=1) of the group excluding the stock itself:
    #   peers' sum of squares  = q - x^2
    #   peers' count           = n - 1
    #   var = (sum_sq - count * mean^2) / (count - 1)

    loo_var = (q - x ** 2 - (n - 1) * loo_mean ** 2) / (n - 2)
    loo_std = np.sqrt(loo_var.clip(lower=0.0))

    valid = (n >= min_group_size) & (loo_std > STD_FLOOR)

    z = np.where(valid, (x - loo_mean) / loo_std.where(valid, 1.0), 0.0)
    df[z_col] = np.clip(z, -Z_CAP, Z_CAP)

    return df