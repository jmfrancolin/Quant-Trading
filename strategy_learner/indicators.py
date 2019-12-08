import pandas as pd
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
from util import get_data, plot_data


def author():
    return 'jfrancolin3'


def get_indicators(df_price, symbol, lookback_window = 10):

    # create a indicators dataframe
    df_indicators = pd.DataFrame()

    # normalize prices
    df_price = df_price / df_price.iloc[0]

    # 10d mean & std
    mean = df_price[symbol].rolling(window = lookback_window).mean()
    std = df_price[symbol].rolling(window = lookback_window).std()

    # Momentum
    df_indicators['MOMENTUM'] = (df_price[symbol] / df_price[symbol].shift(1)) - 1

    # MACD
    mean_12 = df_price[symbol].rolling(window = 12).mean()
    mean_26 = df_price[symbol].rolling(window = 26).mean()
    df_indicators['MACD'] = mean_12 - mean_26

    # Bollinger Bands
    df_indicators['BB_UB'] = mean + 2 * std
    df_indicators['BB_LB']  = mean - 2 * std

    return df_indicators[25:]


def main(time_frame = 'in_sample'):

    # The in sample/development period is January 1, 2008 to December 31 2009
    # The out of sample/testing period is January 1, 2010 to December 31 2011

    # stablish timeframe
    if time_frame == 'in_sample':
        start_date = dt.datetime(2008, 1, 1)
        end_date = dt.datetime(2009, 12, 31)
    else:
        start_date = dt.datetime(2010, 1, 1)
        end_date = dt.datetime(2011, 12, 31)

    dates = pd.date_range(start_date, end_date)

    # create dataframe
    df_price = get_data([symbol], dates)

    # create a indicators dataframe
    df_indicators = get_indicators(df_price)


if __name__ == "__main__":
    main()
