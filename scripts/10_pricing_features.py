import numpy as np
import pandas as pd
import config

# Load the cleaned price panel
prices = pd.read_parquet(config.PRICES_CLEANED)

# sort
prices = prices.sort_values(['ISIN', 'trading_day']).reset_index(drop=True)