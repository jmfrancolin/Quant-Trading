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

    print(symbol)
    pdb.set_trace()

    # dates = pd.date_range(sd, ed)
    # df_price = get_data([symbol], dates)
    # df_indicators = get_indicators(df_price, symbol)



    # # create signals dataframe
    # df_signals = pd.DataFrame(index = df_indicators.index)

    # # MOMENTUM
    # df_signals['MOMENTUM_signal'] = 0
    # df_signals['MOMENTUM_signal'].loc[df_indicators['MOMENTUM'] < 0] = 1
    # df_signals['MOMENTUM_signal'].loc[df_indicators['MOMENTUM'] > 0] = -1

    # # # MACD
    # df_signals['MACD_signal'] = 0
    # df_signals['MACD_signal'].loc[df_indicators['MACD_oscillator'] > 0.0] = 1
    # df_signals['MACD_signal'].loc[df_indicators['MACD_oscillator'] < 0.0] = 0
    # df_signals['MACD_signal'] = df_signals['MACD_signal'].diff()

    # # Bollinger Bands
    # df_signals['bb_signal'] = 0
    # df_signals['bb_signal'].loc[ abs(df_indicators[symbol] - df_indicators['lower_band']) < 0] = 1
    # df_signals['bb_signal'].loc[ abs(df_indicators[symbol] - df_indicators['upper_band']) < 0] = -1

    # df_signals['ensemble_signal'] = df_signals.sum(axis = 1)
    # df_orders = pd.DataFrame(index = df_signals.index, columns = ['Symbol', 'Order', 'Shares'])
    # df_orders['Symbol'] = symbol
    # df_orders['Shares'] = 0

    # holdings = 0
    # for index, row in df_signals.iterrows():
    #     if row['ensemble_signal'] >= 2:

    #         if holdings == -1000:
    #             df_orders.loc[index, 'Order'] = 'BUY'
    #             df_orders.loc[index, 'Shares'] = 2000
    #             holdings += 2000
    #         elif holdings == 0:
    #             df_orders.loc[index, 'Order'] = 'BUY'
    #             df_orders.loc[index, 'Shares'] = 1000
    #             holdings += 1000

    #     elif ['ensemble_signal'] <= -2:

    #         if holdings == 1000:
    #             df_orders.loc[index, 'Order'] = 'SELL'
    #             df_orders.loc[index, 'Shares'] = 2000
    #             holdings -= 2000
    #         elif holdings == 0:
    #             df_orders.loc[index, 'Order'] = 'SELL'
    #             df_orders.loc[index, 'Shares'] = 1000
    #             holdings -= 1000
    #     else:
    #         df_orders.loc[index, 'Order'] = 'HOLD'

    # # df_orders = df_orders.loc[df_orders['Order'] != 'HOLD']
    # df_orders.to_csv('df_orders.csv')
    # df_values = compute_portvals(df_orders = df_orders, start_val = sv, commission = 9.95, impact = 0.05)

    # print(df_values)

    # return df_values


def benchMark(symbol = "AAPL", sd = dt.datetime(2010, 1, 1), \
        ed = dt.datetime(2011, 12, 31), sv = 100000):

    dates = pd.date_range(sd, ed)
    benchmark_df = get_data([symbol], dates)

    benchmark_df = benchmark_df / benchmark_df.iloc[0]

    print(benchmark_df)
    return benchmark_df


def main(time_frame = 'in_sample'):

    # stablish timeframe
    if time_frame == 'in_sample':
        start_date = dt.datetime(2008, 1, 1)
        end_date = dt.datetime(2009, 12, 31)
    else:
        start_date = dt.datetime(2010, 1, 1)
        end_date = dt.datetime(2011, 12, 31)

    # # Manual Rule
    # manual_df = testPolicy(symbol = 'JPM', start_date, end_date)

    # cum_return = manual_df.iloc[-1]
    # std = manual_df.std()
    # mean = manual_df.mean()

    # # print('portval_manual:')
    # portval_manual = [cum_return, std, mean]
    # # print(portval_manual)
    # # pdb.set_trace()


    # plt.subplot(2, 1, 1)
    # plt.title('Manual Rule')
    # plt.ylabel('Return')
    # plt.plot(manual_df.index, manual_df , color='r', label=symbol)

    # std_p2 = portval_manual[2] + 2 * portval_manual[1]
    # std_m2 = portval_manual[2] - 2 * portval_manual[1]
    # plt.axhline(y=std_p2, linestyle='--', color='b', label='mean + 2 sigma')
    # plt.axhline(y=std_m2, linestyle='--', color='b', label='mean - 2 sigma')
    # plt.legend(loc='best')

    # # Benchmark
    # benchmark_df = benchMark(symbol, start_date, end_date)

    # cum_return = benchmark_df[symbol].iloc[-1]
    # std = benchmark_df[symbol].std()
    # mean = benchmark_df[symbol].mean()

    # # print('portval_benchmark')
    # portval_benchmark = [cum_return, std, mean]
    # # print(portval_benchmark)
    # # pdb.set_trace()

    # plt.subplot(2, 1, 2)
    # plt.title('Benchmark')
    # plt.ylabel('Return')
    # plt.plot(benchmark_df.index, benchmark_df[symbol] , color='g', label=symbol)

    # std_p2 = portval_benchmark[2] + 2 * portval_benchmark[1]
    # std_m2 = portval_benchmark[2] - 2 * portval_benchmark[1]
    # plt.axhline(y=std_p2, linestyle='--', color='b', label='mean + 2 sigma')
    # plt.axhline(y=std_m2, linestyle='--', color='b', label='mean - 2 sigma')
    # plt.legend(loc='best')

    # plt.savefig('./Theoretical.png')
    # plt.show()
    # plt.close()


def author(self):
    return 'jfrancolin3'


if __name__ == "__main__":
    main()
