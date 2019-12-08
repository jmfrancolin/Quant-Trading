import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import matplotlib.pyplot as plt
# import pdb

def author():
    return 'jfrancolin3'


def compute_portvals(df_orders, start_val = 1000000, commission = 9.95, impact = 0.005):

    # get a list of unique trade symbols
    symbols = df_orders['Symbol'].unique().tolist()

    # get start and end dates for trade history
    start_date = df_orders.index[0]
    end_date = df_orders.index[-1]

    # create a df with the historical prices from the given symbols and date range
    df_prices = get_data(symbols, pd.date_range(start_date, end_date))

    # append a 'Cash' column to the prices dateframe
    df_prices['Cash'] = 1.0

    # create trades dataframe
    df_trades = df_prices.copy()
    df_trades[:] = 0.0

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

        # sell orders
        elif row.Order == 'SELL':
            # account for market impact
            curr_price = df_prices.loc[row.Index, row.Symbol] * (1 - impact)
            # place orders into trades dataframe
            df_trades.loc[row.Index, row.Symbol] += (-1) * row.Shares
            # update cash column
            df_trades.loc[row.Index, 'Cash'] += row.Shares * curr_price

        # charge commission for the trade
        df_trades.loc[row.Index, 'Cash'] -= commission


    # create holdings dataframe
    df_holdings = df_prices.copy()
    df_holdings[:] = 0.0
    df_holdings.loc[start_date, 'Cash'] = start_val

    # convert trades dataframe to np array and insert a initial cash row
    array_trades = df_trades.values
    array_initial_cash = df_holdings.iloc[0].values
    array_trades = np.insert(array_trades, 0, array_initial_cash, axis = 0)

    # create a temporary dataframe to perform an expanding sum of the rows
    df_temp = pd.DataFrame(data = array_trades)
    df_temp = df_temp.expanding(1).sum()

    # copy over the values from the temp dataframe to the holdings dataframe
    df_holdings[:] = df_temp.iloc[1:].values

    # create a values dataframe
    df_values = df_prices * df_holdings
    df_values = df_values.sum(axis = 1)

    # Normalize
    df_values = df_values / df_values.iloc[0]

    # compute cumulative return
    cum_ret = df_values.iloc[-1] / df_values.iloc[0] - 1

    # compute vector of daily returns
    daily_ret = (df_values / df_values.shift(1)) - 1

    # compute average daily return
    avg_daily_ret = daily_ret.mean()

    # compute standard deviation of daily returns
    std_daily_ret = daily_ret.std()

    # compute sharpe ratio
    sharpe_ratio = np.sqrt(252) * daily_ret.mean() / daily_ret.std()

    # df_values.plot()
    # plt.show()

    return df_values

if __name__ == "__main__":
    pass
