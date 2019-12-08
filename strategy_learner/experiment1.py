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

    # Strategy learner
    learner = sl.StrategyLearner(verbose = True, impact = 0.005)
    sl_port = learner.addEvidence(symbol, sd, ed)

    # Manual strategy
    ms_port = ms.testPolicy(symbol, sd, ed)

    # Benchmark
    bm_port = ms.benchMark(symbol, sd, ed)

    # Strategy Learner v. Benchmark
    plt.plot(sl_port.index, sl_port , color='r', label='Strategy Learner')
    plt.plot(bm_port.index, bm_port , color='g', label='Benchmark')

    plt.title('Strategy Learner v. Benchmark')
    plt.ylabel('Returns')
    plt.legend(loc='lower left')
    plt.savefig('./Strategy Learner v. Benchmark.png')
    plt.close()

    # Strategy Learner v. Manual Strategy
    plt.plot(sl_port.index, sl_port , color='r', label='Strategy Learner')
    plt.plot(ms_port.index, ms_port , color='g', label='Manual Strategy')

    plt.title('Strategy Learner v. Manual Strategy')
    plt.ylabel('Returns')
    plt.legend(loc='lower left')
    plt.savefig('./Strategy Learner v. Manual Strategy.png')
    plt.close()


if __name__=="__main__":
    print("And finally…")
    main()
