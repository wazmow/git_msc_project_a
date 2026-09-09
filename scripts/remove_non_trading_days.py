import pandas as pd
import numpy as np
import pandas_market_calendars as mcal

def remove_non_trading_days(df : pd.DataFrame, datevar : str) -> pd.DataFrame:
    #This script takes a df and compares it to the NYSE pandas market holiday.
    #It removes any dates that are not in the NYSE calendar such as:
    #weekends and NYSE holidays.
    assert str(df[datevar].dtype).startswith('datetime64'),'The date variable must be of type datetime!'
    # get the NYSE calendar
    nyse = mcal.get_calendar('NYSE')

    #pull all the valid dates inbetween the min df date and max df date.
    schedule = nyse.schedule(start_date=df[datevar].min(), end_date=df[datevar].max())

    #Pull just the date rather than full timestamp.
    valid_days = schedule.index


    #Turn the valid_days index into a df
    valid_days_df = pd.DataFrame({'date': valid_days})

    print(f'There are currently {len(df[datevar].unique())} days in your df')
    print(f'There are {len(valid_days)} days in the NYSE schedule')
    print(f'Therefore we remove {len(df[datevar].unique())-len(valid_days_df)} days from the df')

    #return a cleaned pdf by only returning data that has a date in the valid_days df
    df_cleaned = df[df[datevar].isin(valid_days)]
    print(f'df shape after cleaning:{df_cleaned.shape}')

    return df_cleaned

