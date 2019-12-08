"""MC1-P2: Optimize a portfolio.

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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import scipy.optimize as spo
from util import get_data, plot_data

def f(df, allocs):
    df = df * allocs
    port_val = df.sum(axis=1)
    port_val = (port_val / port_val.shift(1)) - 1
    sharpe = np.sqrt(252) * (port_val.mean() / port_val.std())
    return sharpe * -1

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # --------------------------------------------------------------------------
    # Normalize prices
    prices = prices / prices.iloc[0,:]

    # Initialyze allocations
    allocs = np.ones(prices.shape[1]) / prices.shape[1]

    # Compute bounds and constraints
    bounds = [(0.0,1.0) for i in prices.columns]
    constraints = ({ 'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs) })

    # Optimize
    min_result = spo.minimize(f, allocs, args=(prices,), method='SLSQP', \
        constraints=constraints, bounds=bounds, options={'disp':True})

    # Apply new allocations
    allocs = min_result.x
    port_val = (prices * allocs).sum(axis=1)
    port_val = port_val[1:]

    # Generate plot
    if gen_plot == True:
        prices_SPY = prices_SPY / prices_SPY.iloc[0]
        port_val.plot(label='Portfolio')
        prices_SPY.plot(label='SPY')
        plt.title('Daily Portfolio Value and SPY')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='best')
        plt.savefig('Plot.png')
        plt.close()

    # Compute statistics
    cum_return = port_val[-1] - 1
    port_val = (port_val / port_val.shift(1)) - 1
    sharpe = np.sqrt(252) * (port_val.mean() / port_val.std())

    return allocs, cum_return, port_val.mean(), port_val.std(), sharpe

def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2008,6,1)
    end_date = dt.datetime(2009,6,1)
    symbols = ['IBM', 'X', 'GLD', 'JPM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = True)

    # Print statistics
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
