import pandas as pd
import numpy as np

def train_test_split_dates(df: pd.DataFrame, test_size: float = 0.3):
    date_list = sorted(df['date'].unique())
    list_len = len(date_list)
    train_len = int((1-test_size) * list_len)

    train_date_start = date_list[0]
    train_date_end = date_list[train_len+1]

    test_date_start = date_list[train_len+7]
    test_date_end = date_list[-1]
    return {'train_date_start':train_date_start,
            'train_date_end':train_date_end,
            'test_date_start':test_date_start,
            'test_date_end':test_date_end}