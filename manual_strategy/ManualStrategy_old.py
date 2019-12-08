import pandas as pd
import numpy as np
import datetime as dt
import os
import sys
import matplotlib.pyplot as plt
import marketsimcode as ms
import indicators
from util import get_data, plot_data


class ManualStrategy(object):

    def __init__(self):
        self.position = []


    def author(self):
        return 'jfrancolin3'


    def testPolicy(self,symbol = "AAPL", sd=dt.datetime(2010,1,1), \
                   ed=dt.datetime(2011,12,31), sv = 100000):

        lookback = 10
        lookback_day = dt.timedelta(days=lookback)
        dates = pd.date_range(sd-lookback_day*2, ed)

        price= get_data([symbol],dates,addSPY=True, colname = 'Adj Close')
        price = price.drop(columns=['SPY'])
        price = price.fillna(method='ffill')
        price = price.fillna(method='bfill')

        #get first trade day:
        firsttrade_day= price.index[price.index>sd][0]
        price_normed = price/price.loc[firsttrade_day]

        sma = indicators.get_SMA(price_normed,lookback)[firsttrade_day:]
        sma_ratio = indicators.get_SMAratio(price_normed,lookback)[firsttrade_day:]
        bbp = indicators.get_bbp(price_normed,lookback)[2][firsttrade_day:]
        momentum = indicators.get_momentum(price_normed,lookback)[firsttrade_day:]
        macd = indicators.get_MACD(price)[2][firsttrade_day:]
        price = price[firsttrade_day:]

        macd_cross=pd.DataFrame(0,index=price.index,columns=macd.columns)
        macd_cross[macd[symbol]>0]=1
        macd_cross[1:]=macd_cross.diff()
        macd_cross.iloc[0]=0

        position = price.copy()
        position.iloc[:]=np.NaN

        position[(sma_ratio>1.01)&(bbp>1)&(momentum>0.1)&(macd_cross<=0)]= -1
        position[(sma_ratio<0.99)&(bbp<-1)&(momentum<-0.1)&(macd_cross>=0)]= 1

        position.ffill(inplace=True)
        position.fillna(0,inplace=True)

        orders = position.copy()
        orders.iloc[1:] = orders.diff()*1000
        orders.iloc[0] =0
        return orders

    def benchMark(self, symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000):
        dates = pd.date_range(sd, ed)
        df = get_data([symbol],dates)
        df[symbol]=0.0
        df[symbol].iloc[0]=1000
        return df.drop(['SPY'], axis=1)


#%% main function
def test(start_date = dt.datetime(2008,1,1),end_date = dt.datetime(2009,12,31),\
         sv=100000,symbols='JPM'):
    mas = ManualStrategy()
    #get orders for manual strategy
    df_mas = mas.testPolicy(symbols,start_date,end_date,sv=sv)
    #get portfolio for best stragegy
    port_vals_mas =ms.compute_portvals(orders=df_mas,start_val=sv,commission=9.95,\
                                       impact=0.05)
    #properties of best strategy
    port_vals_mas.columns=[symbols]
    port_vals_mas=port_vals_mas[symbols]
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = \
    ms.compute_portfolio(port_vals_mas)

    #get orders for benchmark
    df_bm = mas.benchMark(symbols,start_date,end_date,sv=sv)
    #get portfolio for benchmark
    port_vals_bm = ms.compute_portvals(orders=df_bm,start_val=sv,commission=9.95,
                                       impact=0.05)
    port_vals_bm.columns=[symbols]
    port_vals_bm = port_vals_bm[symbols]
    cum_ret_bm, avg_daily_ret_bm, std_daily_ret_bm, sharpe_ratio_bm = \
        ms.compute_portfolio(port_vals_bm)

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of benchmark : {}".format(sharpe_ratio_bm)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of benchmark : {}".format(cum_ret_bm)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of benchmark : {}".format(std_daily_ret_bm)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of benchmark : {}".format(avg_daily_ret_bm)
    print
    print "Final Portfolio Value of Fund (Normalized): {}".\
    format(port_vals_mas[-1]/port_vals_mas[0])
    print "Final Portfolio Value of benchmark (Normalized): {}".\
    format(port_vals_bm[-1]/port_vals_bm[0])
    print

    port_vals_mas_norm = port_vals_mas/port_vals_mas[0]
    port_vals_bm_norm = port_vals_bm/port_vals_bm[0]

    return df_mas,port_vals_mas_norm,port_vals_bm_norm


def main():
    #insample
    df_in,mas_in,bm_in=test(start_date = dt.datetime(2008,1,1),end_date = dt.datetime(2009,12,31),\
         sv=100000,symbols='JPM')
    df_out,mas_out,bm_out=test(start_date = dt.datetime(2010,1,1),end_date=dt.datetime(2011,12,31))

    long_in=df_in.index[df_in['JPM']>0].tolist()
    short_in=df_in.index[df_in['JPM']<0].tolist()
    #%%
    price_in = get_data(['JPM'],pd.date_range(dt.datetime(2008,1,1),dt.datetime\
                        (2009,12,31)),addSPY=True, colname = 'Adj Close')

    plt.figure(figsize=(10,5))
    plt.subplot(211)
    plt.plot(price_in.index,price_in['JPM'],'blue',label='price')
    ymin,ymax = plt.ylim()
    plt.vlines(x=long_in,ymin=ymin,ymax=ymax,color='blue',linestyle=['-.'])
    plt.vlines(x=short_in,ymin=ymin,ymax=ymax,color='black',linestyle=['-.'])
    plt.title('In sample: manual strategy vs benchmark')
    plt.ylabel('price')
    plt.subplot(212)
    plt.plot(mas_in.index, mas_in, 'r',label='manual strategy')
    plt.plot(bm_in.index, bm_in, 'g',label='benchmark')
    ymin,ymax = plt.ylim()
    plt.vlines(x=long_in,ymin=ymin,ymax=ymax,color='blue',linestyle=['-.'])
    plt.vlines(x=short_in,ymin=ymin,ymax=ymax,color='black',linestyle=['-.'])

    plt.xlabel('date')
    plt.ylabel('normalized portfolio')

    plt.legend()
    plt.savefig('./part2_insample.png')
    plt.show()
    plt.close

    price_out = get_data(['JPM'],pd.date_range(dt.datetime(2010,1,1),dt.datetime\
                        (2011,12,31)),addSPY=True, colname = 'Adj Close')

    long_out=df_out.index[df_out['JPM']>0].tolist()
    short_out=df_out.index[df_out['JPM']<0].tolist()

    plt.figure(figsize=(10,5))
    plt.subplot(211)
    plt.plot(price_out.index,price_out['JPM'],'blue',label='price')
    ymin,ymax = plt.ylim()
    plt.vlines(x=long_out,ymin=ymin,ymax=ymax,color='blue',linestyle=['-.'])
    plt.vlines(x=short_out,ymin=ymin,ymax=ymax,color='black',linestyle=['-.'])
    plt.title('Out sample: manual strategy vs benchmark')
    plt.ylabel('price')
    plt.subplot(212)
    plt.plot(mas_out.index, mas_out, 'r',label='manual strategy')
    plt.plot(bm_out.index, bm_out, 'g',label='benchmark')
    ymin,ymax = plt.ylim()
    plt.vlines(x=long_out,ymin=ymin,ymax=ymax,color='blue',linestyle=['-.'])
    plt.vlines(x=short_out,ymin=ymin,ymax=ymax,color='black',linestyle=['-.'])

    plt.xlabel('date')
    plt.ylabel('normalized portfolio')
    plt.legend()
    plt.savefig('./part2_outsample.png')
    plt.show()
    plt.close

if __name__ == "__main__":
    main()
