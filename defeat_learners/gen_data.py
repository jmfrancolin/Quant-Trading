"""
template for generating data to fool learners (c) 2016 Tucker Balch
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

import numpy as np
import math

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# scp -r ML4T_2019Fall/defeat_learners/gen_data.py jfrancolin3@buffet01.cc.gatech.edu:~/ML4T_2019Fall/defeat_learners

# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best4LinReg(seed = 1489683273):

    # Linear Regression is an appropriate method to model a system that have an
    # underlining behavior that could be mathematically described with an linear
    # equation (perhaps with  multiple dimensions). Hence, we will artificially
    # fabricate such behavior using random data.

    # seed the random number generator
    np.random.seed(seed)

    # generate X
    dim_0 = np.linspace(0, 999, num = (10))
    X = np.vstack((dim_0, dim_0)).transpose()

    # generate Y
    Y = np.random.randint(1000, size = (1000))

    for ii in range(len(Y)):
      Y[ii] = np.sum(Y[ii:])

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection = '3d')
    # ax.scatter(dim_0, dim_1, Y)
    # plt.show()

    return X, Y


def best4DT(seed = 1489683273):

    # Decision Trees is best used to divide a plane (of n-dimentions) into
    # discrete regions. Is an appropriate method to model a system that with
    # non-linearly separable data. Hence, we will artificially fabricate such
    # behavior using random data.

    # seed the random number generator
    np.random.seed(seed)

    # generate X
    X = np.random.randint(1000, size = (1000, 2))

    # fold it once
    mask_dim0_left = (X[:, 0] >= 250) & (X[:, 0] < 500)
    X[:, 0][mask_dim0_left] = X[:, 0][mask_dim0_left] - 250

    # fold it twice
    mask_dim0_right = (X[:, 0] >= 500) & (X[:, 0] < 750)
    X[:, 0][mask_dim0_right] = X[:, 0][mask_dim0_right] + 250

    # three times
    mask_dim1_down = (X[:, 1] >= 250) & (X[:, 1] < 500)
    X[:, 1][mask_dim1_down] = X[:, 1][mask_dim1_down] - 250

    # and four times is the charm
    mask_dim1_up = (X[:, 1] >= 500) & (X[:, 1] < 750)
    X[:, 1][mask_dim1_up] = X[:, 1][mask_dim1_up] + 250

    # generate Y
    Y = np.linspace(0, 999, num = 1000)

    # create masks for map Y into separable X data
    mask_y_left = (X[:, 0] <= 250)
    mask_y_right = (X[:, 0] >= 750)

    mask_y_down = (X[:, 1] <= 250)
    mask_y_up = (X[:, 1] >= 750)

    Y[mask_y_left & mask_y_down] = 0
    Y[mask_y_left & mask_y_up] = 1
    Y[mask_y_right & mask_y_down] = 2
    Y[mask_y_right & mask_y_up] = 3

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection = '3d')
    # ax.scatter(X[:, 0], X[:, 1], Y)
    # plt.show()

    return X, Y


def author():
    return 'jfrancolin3' #Change this to your user ID

if __name__=="__main__":
    print("they call me Tim.")
    # best4LinReg()
    # best4DT()
