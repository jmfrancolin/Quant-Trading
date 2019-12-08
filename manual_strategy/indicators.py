import pandas as pd
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
# from matplotlib.finance import candlestick_ohlc
from util import get_data, plot_data


def author():
    return 'jfrancolin3'


def get_indicators(df_price):

    # define a lookback window
    lookback_window = 10

    # create a indicators dataframe
    df_indicators = df_price.copy()

    # normalize prices
    df_indicators = df_indicators / df_indicators.iloc[0,:]

    # 10d mean & std
    df_indicators['mean'] = df_indicators['JPM'].rolling(window = lookback_window).mean()
    df_indicators['std'] = df_indicators['JPM'].rolling(window = lookback_window).std()

    # Momentum
    df_indicators['momentum'] = (df_indicators['JPM'] / df_indicators['JPM'].shift(lookback_window)) - 1

    # SMA
    df_indicators['sma'] = df_indicators['mean']

    # MACD
    df_indicators['mean_12'] = df_indicators['JPM'].rolling(window = 12, min_periods = 1).mean()
    df_indicators['mean_26'] = df_indicators['JPM'].rolling(window = 26, min_periods = 1).mean()
    df_indicators['macd_oscillator'] = df_indicators['mean_12'] - df_indicators['mean_26']

    # Bollinger Bands
    df_indicators['mid_band'] = df_indicators['mean']
    df_indicators['upper_band'] = df_indicators['mean'] + 2 * df_indicators['std']
    df_indicators['lower_band']  = df_indicators['mean'] - 2 * df_indicators['std']

    return df_indicators


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
    df_price = get_data(['JPM'], dates)

    # create a indicators dataframe
    df_indicators = get_indicators(df_price)

    # PLOTS

    # Momentum
    plt.plot(df_indicators.index, df_indicators['JPM'] , 'g', label='JPM')
    plt.plot(df_indicators.index, df_indicators['momentum'].diff(), 'r', label='d/dt Momentum')
    plt.title('Price & Momentum First Derivative')
    plt.legend()
    plt.savefig('./Momentum.png')
    # plt.show()
    plt.close()

    # SMA
    plt.plot(df_indicators.index, df_indicators['JPM'], 'g', label='JPM')
    plt.plot(df_indicators.index, df_indicators['mean'], 'r', label='SMA')
    plt.ylabel('Price')
    plt.title('Price & SMA')
    plt.legend()
    plt.savefig('./SMA.png')
    # plt.show()
    plt.close()

    # MACD
    plt.subplot(2, 1, 1)
    plt.title('Price & MACD Ocilator')
    plt.ylabel('Price')
    plt.plot(df_indicators.index, df_indicators['JPM'] , 'g', label='JPM')
    plt.subplot(2, 1, 2)
    plt.bar(df_indicators.index, df_indicators['macd_oscillator'], color='r', label='Oscillator')
    plt.savefig('./MACD.png')
    # plt.show()
    plt.close()

    # Bollinger Bands
    plt.plot(df_indicators.index, df_indicators['JPM'] , 'g', label='JPM')
    plt.plot(df_indicators.index, df_indicators['upper_band'], 'r', label='Upper Band')
    plt.plot(df_indicators.index, df_indicators['lower_band'], 'b', label='Lower Band')
    plt.ylabel('Price')
    plt.title('Price & Momentum Bollinger Bands')
    plt.legend()
    plt.savefig('./BB.png')
    # plt.show()
    plt.close()


if __name__ == "__main__":
    main()
