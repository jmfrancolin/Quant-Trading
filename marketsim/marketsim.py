"""MC2-P1: Market simulator.

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Joao Matheus Nascimento Francolin
GT User ID: jfrancolin3
GT ID: 903207758
"""

# PYTHONPATH=..:. python marketsim.py

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
# import pdb

def author():
    return 'jfrancolin3'

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission = 9.95, impact = 0.005):

    # create orders dataframe
    df_orders = pd.read_csv(orders_file, index_col = 'Date', parse_dates = True)
    df_orders.sort_index(inplace = True)

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

    return df_values

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-short.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

if __name__ == "__main__":
    test_code()
