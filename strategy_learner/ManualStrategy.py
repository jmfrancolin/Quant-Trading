import pandas as pd
import numpy as np
import datetime as dt
import os
import sys
import matplotlib.pyplot as plt
from indicators import get_indicators
from marketsimcode import compute_portvals
from util import get_data, plot_data
import pdb


def testPolicy(symbol = "AAPL", sd = dt.datetime(2010, 1, 1), \
    ed = dt.datetime(2011, 12, 31), sv = 100000):

    dates = pd.date_range(sd, ed)
    df_price = get_data([symbol], dates)
    df_price = df_price / df_price.iloc[0]
    df_indicators = get_indicators(df_price, symbol)

    look_back_window = len(df_price) - len(df_indicators)
    df_price = df_price[look_back_window:]

    # create signals dataframe
    df_signals = pd.DataFrame(index = df_indicators.index)

    # MOMENTUM
    df_signals['MOMENTUM'] = 0
    df_signals['MOMENTUM'].loc[df_indicators['MOMENTUM'] > 0.8] = 1
    df_signals['MOMENTUM'].loc[df_indicators['MOMENTUM'] < 0.5] = -1

    # MACD
    df_signals['MACD'] = 0
    df_signals['MACD'].loc[df_indicators['MACD'] > 0.5] = 1
    df_signals['MACD'].loc[df_indicators['MACD'] < -0.5] = -1

    # Bollinger Bands
    df_signals['BB'] = 0
    df_signals['BB'].loc[df_price[symbol] < df_indicators['BB_LB']] = 1
    df_signals['BB'].loc[df_price[symbol] > df_indicators['BB_UB']] = -1

    df_signals['ensemble_signal'] = df_signals.sum(axis = 1)
    df_orders = pd.DataFrame(index = df_signals.index, columns = ['Symbol', 'Order', 'Shares'])
    df_orders['Symbol'] = symbol
    df_orders['Shares'] = 0

    holdings = 0
    for index, row in df_signals.iterrows():

        if row['ensemble_signal'] >= 2:
            if holdings == -1000:
                df_orders.loc[index, 'Order'] = 'BUY'
                df_orders.loc[index, 'Shares'] = 2000
                holdings += 2000
            elif holdings == 0:
                df_orders.loc[index, 'Order'] = 'BUY'
                df_orders.loc[index, 'Shares'] = 1000
                holdings += 1000

        elif row['ensemble_signal'] <= -2:
            if holdings == 1000:
                df_orders.loc[index, 'Order'] = 'SELL'
                df_orders.loc[index, 'Shares'] = 2000
                holdings -= 2000
            elif holdings == 0:
                df_orders.loc[index, 'Order'] = 'SELL'
                df_orders.loc[index, 'Shares'] = 1000
                holdings -= 1000

        elif row['ensemble_signal'] == 0:
            if holdings == 1000:
                df_orders.loc[index, 'Order'] = 'SELL'
                df_orders.loc[index, 'Shares'] = 1000
                holdings -= 1000

            elif holdings == -1000:
                df_orders.loc[index, 'Order'] = 'BUY'
                df_orders.loc[index, 'Shares'] = 1000
                holdings += 1000


    df_port = compute_portvals(df_orders = df_orders, start_val = sv, commission = 9.95, impact = 0.05)

    return df_port


def benchMark(symbol = "AAPL", sd = dt.datetime(2010, 1, 1), \
        ed = dt.datetime(2011, 12, 31), sv = 100000):

    dates = pd.date_range(sd, ed)
    df_price = get_data([symbol], dates)
    df_price = df_price / df_price.iloc[0]
    df_indicators = get_indicators(df_price, symbol)

    df_indicators = get_indicators(df_price, symbol)
    df_orders = pd.DataFrame(index = df_indicators.index, columns = ['Symbol', 'Order', 'Shares'])
    df_orders['Symbol'] = symbol
    df_orders['Shares'] = 0

    df_orders.loc[df_orders.index[0], 'Order'] = 'BUY'
    df_orders.loc[df_orders.index[0], 'Shares'] = 1000

    df_port = compute_portvals(df_orders = df_orders, start_val = sv, commission = 9.95, impact = 0.05)

    return df_port


def main():

    #in_sample'
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)

    symbol = 'JPM'

    # Manual strategy
    df_ms = testPolicy(symbol, start_date, end_date)
    # Benchmark
    df_bm = benchMark(symbol, start_date, end_date)


    plt.plot(df_ms.index, df_ms , color='r', label='Manual Strategy')
    plt.plot(df_bm.index, df_bm , color='g', label='Benchmark')

    plt.title('Manual Strategy v. Benchmark')
    plt.ylabel('Returns')
    plt.legend(loc='lower left')
    plt.savefig('./Manual Strategy v. Benchmark.png')
    plt.close()


def author(self):
    return 'jfrancolin3'


if __name__ == "__main__":
    main()
