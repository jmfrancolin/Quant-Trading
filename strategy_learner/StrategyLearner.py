"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

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

Student Name: Tucker Balch (replace with your name)
GT User ID: tb34 (replace with your User ID)
GT ID: 900897987 (replace with your GT ID)
"""

import datetime as dt
import pandas as pd
import util as ut
import random
import QLearner as ql
from indicators import get_indicators
from util import get_data
import numpy as np
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
# import pdb


class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0):
        self.verbose = verbose
        self.impact = impact

        self.mean = {}
        self.std = {}
        self.bins = {}

        self.impact = impact

        self.holdings = 0
        self.learner = ql.QLearner(num_states = 10000, num_actions = 3, alpha = 0.2, \
            gamma = 0.9, rar = 0.5, radr = 0.99, dyna = 0, verbose = False)


    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), sv = 1000000):

        # get data range
        dates = pd.date_range(sd, ed)
        # get historical prices
        df_price = get_data([symbol], dates)
        # compute indicators from historical prices
        df_indicators = get_indicators(df_price, symbol)
        # initialize a dataframe for orders
        df_orders = pd.DataFrame(index = df_indicators.index, \
            columns = ['Symbol', 'Order', 'Shares'])
        df_orders['Symbol'] = symbol

        # discretize indicators of trainning set
        # in order to train, we must look ahead in the data to construct
        # proper sized bins
        self.discretize_train_indicators(df_indicators)
        # compute daily returns of trainning set
        daily_returns = df_price[symbol] / df_price[symbol].shift(1) - 1

        # control variables
        iteration = 0
        converged = False
        while not converged or iteration < 10:
            # reset orders dataframe
            df_orders = pd.DataFrame(index = df_indicators.index, \
            columns = ['Symbol', 'Order', 'Shares'])
            df_orders['Symbol'] = symbol
            # reset holdings
            self.holdings = 0

            # query first state
            state = df_indicators.iloc[0].values
            # construct a single integer state discretized indicator
            state = int(f"{state[0]}{state[1]}{state[2]}{state[3]}")
            # get initial action
            action = self.learner.querysetstate(state)
            # perform first trade
            self.perform_trade(df_orders, action, df_indicators.index.values[0])

            # loop through indicators database
            for day, _ in df_indicators[1:].T.iteritems():
                # get reward based on current holdings and markets return
                reward = self.get_reward(day, daily_returns[day])
                # query state from indicators
                state = df_indicators.loc[day].values
                # construct a single integer state discretized indicator
                state = int(f"{state[0]}{state[1]}{state[2]}{state[3]}")
                # query action from learner
                action = self.learner.query(state, reward)
                # perform trade
                self.perform_trade(df_orders, action, day)

            # creare a copy of the orders dataframe to test convergence
            df_orders_copy = df_orders.copy()

            # update control variables
            iteration += 1
            if df_orders.equals(df_orders_copy):
                converged = True

        # used for experiment comparaion code
        if self.verbose:
            df_orders = df_orders[pd.notnull(df_orders['Shares'])]
            df_port = compute_portvals(df_orders = df_orders, start_val = sv, commission = 0, impact = self.impact)
            return df_port
            # df_port.plot()
            # plt.show()


    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), sv = 1000000):

        # get data range
        dates = pd.date_range(sd, ed)
        # get historical prices
        df_price = get_data([symbol], dates)
        # compute indicators from historical prices
        df_indicators = get_indicators(df_price, symbol)

        # create orders dataframe
        df_orders = pd.DataFrame(index = df_indicators.index, \
        columns = ['Symbol', 'Order', 'Shares'])
        df_orders['Symbol'] = symbol
        # reset holdings
        self.holdings = 0

        # query first state
        state = df_indicators.iloc[0].values
        # discretize indicators of testing set
        # no lookahead allowed
        state = self.discretize_indicators(state)
        # construct a single integer state discretized indicator
        state = int(f"{state[0]}{state[1]}{state[2]}{state[3]}")
        # query action based on model
        action = self.learner.querysetstate(state)
        # perform first trade
        self.perform_trade(df_orders, action, df_indicators.index.values[0])
        # save last date
        trade_date = df_indicators.index.values[0]

        # loop through indicators database
        for day, _ in df_indicators[1:].T.iteritems():

            # compute daily_return based on market price diff form trade_date to next date
            daily_return = df_price.loc[day, symbol] / df_price.loc[trade_date, symbol] - 1
            # get reward based on current holdings and markets return
            reward = self.get_reward(day, daily_return)
            # query state from indicators
            state = df_indicators.loc[day].values
            # discretize indicators of testing set
            state = self.discretize_indicators(state)
            # construct a single integer state discretized indicator
            state = int(f"{state[0]}{state[1]}{state[2]}{state[3]}")
            # query action based on model
            action = self.learner.querysetstate(state)
            # perform first trade
            self.perform_trade(df_orders, action, day)
            # update date after trade is performed
            trade_date = day

        df_orders = df_orders[pd.notnull(df_orders['Shares'])]
        df_port = compute_portvals(df_orders = df_orders, start_val = sv, commission = 0, impact = self.impact)
        # df_port.plot()
        # plt.show()

        # Formating for autograder
        df_trades = df_orders.copy()
        df_trades['Order'].loc[df_trades['Order'] == 'BUY'] = 1
        df_trades['Order'].loc[df_trades['Order'] == 'SELL'] = -1
        df_trades['Symbol'] = df_trades['Order'] * df_trades['Shares']
        trades = df_trades['Symbol']
        trades = pd.DataFrame(trades)

        if self.verbose: print(type(trades)) # it better be a DataFrame!
        if self.verbose: print(trades)
        if self.verbose: print(df_price)
        return trades

    def perform_trade(self, df_orders, action, day):

        # LONG
        if action == 0:
            if self.holdings == -1000:
                df_orders.loc[day, 'Order'] = 'BUY'
                df_orders.loc[day, 'Shares'] = 2000
                self.holdings += 2000

            elif self.holdings == 0:
                df_orders.loc[day, 'Order'] = 'BUY'
                df_orders.loc[day, 'Shares'] = 1000
                self.holdings += 1000

        # SHORT
        elif action == 1:
            if self.holdings == 1000:
                df_orders.loc[day, 'Order'] = 'SELL'
                df_orders.loc[day, 'Shares'] = 2000
                self.holdings -= 2000

            elif self.holdings == 0:
                df_orders.loc[day, 'Order'] = 'SELL'
                df_orders.loc[day, 'Shares'] = 1000
                self.holdings -= 1000

        # HOLD
        elif action == 2:
            if self.holdings == 1000:
                df_orders.loc[day, 'Order'] = 'SELL'
                df_orders.loc[day, 'Shares'] = 1000
                self.holdings -= 1000

            elif self.holdings == -1000:
                df_orders.loc[day, 'Order'] = 'BUY'
                df_orders.loc[day, 'Shares'] = 1000
                self.holdings += 1000


    def get_reward(self, day, daily_return):

        # HOLD
        reward = 0

        # LONG
        if self.holdings > 0:
            reward = daily_return
        # SHORT
        elif self.holdings < 0:
            reward = (-1) * daily_return

        return reward


    def discretize_train_indicators(self, df_train, steps = 10):

        for col in df_train.columns:
            # normalize data
            self.mean[col] = df_train[col].mean()
            self.std[col] = df_train[col].std()
            df_train[col] = (df_train[col] - self.mean[col]) / self.std[col]

            # sort data
            step_size = df_train[col].size / steps
            sorted_data = np.sort(df_train[col])

            # get theshold
            threshold = np.empty([steps])
            for i in range(steps):
                threshold[i] = sorted_data[round((i + 1) * step_size) - 1]

            # discretize
            digitized = np.digitize(df_train[col], threshold)
            digitized[digitized == 10] = 9

            # save data
            df_train[col] = digitized
            self.bins[col] = threshold


    def discretize_indicators(self, data, steps = 10):

        # initialize digitized array
        digitized = np.empty([len(data)])

        for index in range(len(data)):
            # normalize data
            mean = list(self.mean.values())[index]
            std = list(self.std.values())[index]
            data[index] = (data[index] - mean) / std

            # get threshold computed on training data
            threshold = list(self.bins.values())[index]

            # discretize
            digitized[index] = np.digitize(data[index], threshold)

        digitized[digitized == 10] = 9
        return digitized.astype(int)


    def author(self):
        return 'jfrancolin3'


if __name__=="__main__":
    print("One does not simply think up a strategy")
