import pandas as pd
import numpy as np
import datetime as dt
import os
import sys
import matplotlib.pyplot as plt
from util import get_data, plot_data
from marketsimcode import compute_portvals
import pdb

def author(self):
    return 'jfrancolin3'


def testPolicy(symbol = "AAPL", sd = dt.datetime(2010, 1, 1), \
        ed = dt.datetime(2011, 12, 31), sv = 100000):

    dates = pd.date_range(sd, ed)
    optimum_df = get_data([symbol], dates)
    optimum_df = optimum_df / optimum_df.iloc[0]
    optimum_df['daily_ret'] = optimum_df[symbol].diff().shift(-1)

    df_orders = pd.DataFrame(index = optimum_df.index, columns = ['Symbol', 'Order', 'Shares'])
    df_orders['Symbol'] = symbol
    df_orders['Shares'] = 0

    holdings = 0
    for index, row in optimum_df.iterrows():

        if optimum_df.loc[index, 'daily_ret'] > 0:

            if holdings == -1000:
                df_orders.loc[index, 'Order'] = 'BUY'
                df_orders.loc[index, 'Shares'] = 2000
                holdings += 2000
            elif holdings == 0:
                df_orders.loc[index, 'Order'] = 'BUY'
                df_orders.loc[index, 'Shares'] = 1000
                holdings += 1000

        elif optimum_df.loc[index, 'daily_ret'] < 0:

            if holdings == 1000:
                df_orders.loc[index, 'Order'] = 'SELL'
                df_orders.loc[index, 'Shares'] = 2000
                holdings -= 2000
            elif holdings == 0:
                df_orders.loc[index, 'Order'] = 'SELL'
                df_orders.loc[index, 'Shares'] = 1000
                holdings -= 1000
        else:
            df_orders.loc[index, 'Order'] = 'HOLD'

    df_orders.to_csv('df_optimum_orders.csv')
    df_values = compute_portvals(df_orders = df_orders, start_val = sv, commission = 0, impact = 0)

    return df_values

def benchMark(symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000):

    dates = pd.date_range(sd, ed)
    benchmark_df = get_data([symbol], dates)

    benchmark_df = benchmark_df / benchmark_df.iloc[0]

    print(benchmark_df)
    return benchmark_df


def main(time_frame = 'out_sample'):

    symbol = 'IBM'

    # stablish timeframe
    if time_frame == 'in_sample':
        start_date = dt.datetime(2008, 1, 1)
        end_date = dt.datetime(2009, 12, 31)
    else:
        start_date = dt.datetime(2010, 1, 1)
        end_date = dt.datetime(2011, 12, 31)

    # Optimum
    optimal_df = testPolicy(symbol, start_date, end_date)

    cum_return = optimal_df.iloc[-1]
    std = optimal_df.std()
    mean = optimal_df.mean()
    portval_optimal = [cum_return, std, mean]

    # print(portval_optimal)
    # pdb.set_trace()

    plt.subplot(2, 1, 1)
    plt.title('Theoretical Optimum')
    plt.ylabel('Return')
    plt.plot(optimal_df.index, optimal_df , color='r', label=symbol)

    std_p2 = portval_optimal[2] + 2 * portval_optimal[1]
    std_m2 = portval_optimal[2] - 2 * portval_optimal[1]
    plt.axhline(y=std_p2, linestyle='--', color='b', label='mean + 2 sigma')
    plt.axhline(y=std_m2, linestyle='--', color='b', label='mean - 2 sigma')
    plt.legend(loc='best')

    # Benchmark
    benchmark_df = benchMark(symbol, start_date, end_date)

    cum_return = benchmark_df.iloc[-1]
    std = benchmark_df[symbol].std()
    mean = benchmark_df[symbol].mean()
    portval_benchmark = [cum_return, std, mean]

    # print(portval_benchmark)
    # pdb.set_trace()

    plt.subplot(2, 1, 2)
    plt.title('Benchmark')
    plt.ylabel('Return')
    plt.plot(benchmark_df.index, benchmark_df[symbol] , color='g', label=symbol)

    std_p2 = portval_benchmark[2] + 2 * portval_benchmark[1]
    std_m2 = portval_benchmark[2] - 2 * portval_benchmark[1]
    plt.axhline(y=std_p2, linestyle='--', color='b', label='mean + 2 sigma')
    plt.axhline(y=std_m2, linestyle='--', color='b', label='mean - 2 sigma')
    plt.legend(loc='best')

    plt.savefig('./Theoretical.png')
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
