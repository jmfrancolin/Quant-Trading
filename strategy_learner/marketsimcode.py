import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import matplotlib.pyplot as plt
import pdb

def author():
    return 'jfrancolin3'


def compute_portvals(df_orders, start_val = 1000000, commission = 9.95, impact = 0.005):

    # get stock symbol
    symbol = df_orders['Symbol'].unique()[0]

    # get start and end dates for trade history
    start_date = df_orders.index[0]
    end_date = df_orders.index[-1]

    # create a df with the historical prices from the given symbols and date range
    df_prices = get_data([symbol], pd.date_range(start_date, end_date))

    # append a 'Cash' column to the prices dateframe
    df_prices['Cash'] = 1.0

    # create trades dataframe
    df_trades = df_prices.copy()
    df_trades[:] = 0.0

    trade = False
    # iterate through orders dataframe
    for row in df_orders.itertuples():

        # buy orders
        if row.Order == 'BUY':
            # account for market impact
            curr_price = df_prices.loc[row.Index, row.Symbol] * (1 + impact)
            # place orders into trades dataframe
            df_trades.loc[row.Index, row.Symbol] += row.Shares
            # update cash column
            df_trades.loc[row.Index, 'Cash'] += (-1) * row.Shares * curr_price
            trade = True

        # sell orders
        elif row.Order == 'SELL':
            # account for market impact
            curr_price = df_prices.loc[row.Index, row.Symbol] * (1 - impact)
            # place orders into trades dataframe
            df_trades.loc[row.Index, row.Symbol] += (-1) * row.Shares
            # update cash column
            df_trades.loc[row.Index, 'Cash'] += row.Shares * curr_price
            trade = True

        if trade == True:
            # charge commission for the trade
            df_trades.loc[row.Index, 'Cash'] -= commission

    # crete holdings dataframe
    df_holdings = df_trades.copy()
    df_holdings[:] = 0
    df_holdings.loc[start_date, 'Cash'] += start_val
    df_holdings.iloc[0] = df_trades.iloc[0] + df_holdings.iloc[0]

    for i in range(1, len(df_trades)):
        df_holdings.iloc[i] = df_trades.iloc[i] + df_holdings.iloc[i - 1]


    # create a values dataframe
    df_port = df_prices * df_holdings
    df_port = df_port.sum(axis = 1)

    # Normalize
    df_port = df_port / df_port.iloc[0]

    # compute cumulative return
    cum_ret = df_port.iloc[-1] / df_port.iloc[0] - 1

    # compute vector of daily returns
    daily_ret = (df_port / df_port.shift(1)) - 1

    # compute average daily return
    avg_daily_ret = daily_ret.mean()

    # compute standard deviation of daily returns
    std_daily_ret = daily_ret.std()

    # compute sharpe ratio
    sharpe_ratio = np.sqrt(252) * daily_ret.mean() / daily_ret.std()

    # df_port.plot()
    # plt.show()

    return df_port

if __name__ == "__main__":
    pass
