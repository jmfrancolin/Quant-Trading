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
import StrategyLearner as sl
import ManualStrategy as ms
# import pdb

def author():
    return 'jfrancolin3'

def main():

    # in_sample
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    symbol = 'JPM'

    _returns = []
    num_trades = []

    for impact in np.arange(0, 0.1, 0.005):
        learner = sl.StrategyLearner(verbose = True, impact = impact)
        sl_port = learner.addEvidence(symbol, sd, ed)
        df_trades = learner.testPolicy(symbol, sd, ed)

        num_trades.append(len(df_trades))
        _returns.append(sl_port[-1])

    plt.plot(np.arange(0, 0.1, 0.005), num_trades)
    plt.title('Number of Trades v. Impact')
    plt.ylabel('Number of Trades')
    plt.xlabel('Impact')
    plt.savefig('./Number of Trades v. Impact.png')
    plt.close()

    plt.plot(np.arange(0, 0.1, 0.005), _returns)
    plt.title('Cumulative Returns v. Impact')
    plt.ylabel('Cumulative Returns')
    plt.xlabel('Impact')
    plt.savefig('./Cumulative Returns v. Impact.png')
    plt.close()

if __name__=="__main__":
    print("And finally (2)…")
    main()
