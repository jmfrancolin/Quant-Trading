"""Assess a betting strategy.

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
import matplotlib.pyplot as plt

def author():
    return 'jfrancolin3'

def gtid():
    return 903207758

def get_spin_result(win_prob):
    result = False
    if np.random.random() <= win_prob:
        result = True

    return result

def test_code():
    win_prob = 18 / 38 # set appropriately to the probability of a win
    np.random.seed(gtid()) # do this only once
    # print(get_spin_result(win_prob)) # test the roulette spin

    # add your code here to implement the experiments
    experiment_1(win_prob)
    experiment_2(win_prob)

def strategy_simulator(max_trials, win_prob):

    winnings = np.zeros(1000, int)
    trial_counter = 0

    while trial_counter < max_trials:
        cum_sum = 0
        index = 0
        new_row = np.zeros(1000, int)

        while cum_sum < 80:
            trial_win = False
            bet = 1
            while not trial_win:
                trial_win = get_spin_result(win_prob)
                if trial_win:
                    cum_sum += bet
                else:
                    cum_sum -= bet
                    bet *= 2

                new_row[index] = cum_sum
                index += 1

        new_row[index:new_row.size] = new_row[index - 1]
        winnings = np.vstack((winnings, new_row))
        trial_counter += 1

    return winnings[1:,:].T

def experiment_1(win_prob):

    winnings = strategy_simulator(10, win_prob)

    # Figure 1
    plt.plot(winnings)
    plt.axis([0, 300, -256, 100])
    plt.title('Figure 1')
    plt.xlabel('Roll Number')
    plt.ylabel('Cumulative Sum ($)')
    plt.savefig('Figure_1.png')
    plt.close()

    winnings = strategy_simulator(1000, win_prob)
    mean = winnings.mean(axis = 1)
    median = np.median(winnings, axis = 1)
    std = winnings.std(axis = 1)

    # Figure 2
    plt.plot(mean, label='Mean')
    plt.plot(mean + std, label='Mean + Std')
    plt.plot(mean - std, label='Mean - Std')
    plt.axis([0, 300, -256, 100])
    plt.title('Figure 2')
    plt.xlabel('Roll Number')
    plt.ylabel('Cumulative Sum ($)')
    plt.legend(loc='lower right')
    plt.savefig('Figure_2.png')
    plt.close()

    # Figure 3
    plt.plot(median, label='Median')
    plt.plot(median + std, label='Median + Std')
    plt.plot(median - std, label='Median - Std')
    plt.axis([0, 300, -256, 100])
    plt.title('Figure 3')
    plt.xlabel('Roll Number')
    plt.ylabel('Cumulative Sum ($)')
    plt.legend(loc='lower right')
    plt.savefig('Figure_3.png')
    plt.close()

    # Questions
    print('Q1. ', sum(winnings[-1,:] == 80) / 1000)
    print('Q2. ', sum(winnings[-1,:]) / 1000)
    plt.title('Q3.')
    plt.plot(std, label='Standard Deviation')
    plt.xlabel('Roll Number')
    plt.legend(loc='lower right')
    plt.savefig('Q3.png')
    plt.close()

def realistic_simulator(max_trials, win_prob):

    winnings = np.zeros(1000, int)
    trial_counter = 0

    while trial_counter < max_trials:
        cum_sum = 0
        index = 0
        bank = 256
        new_row = np.zeros(1000, int)

        while (cum_sum < 80)  and (cum_sum > -256):
            trial_win = False
            bet = 1
            while not trial_win:
                trial_win = get_spin_result(win_prob)
                if trial_win:
                    cum_sum += bet
                else:
                    cum_sum -= bet
                    bet *= 2

                    if bet > (bank + cum_sum):
                        bet = bank + cum_sum

                new_row[index] = cum_sum
                index += 1

                if cum_sum <= -256:
                    break

        new_row[index:new_row.size] = new_row[index - 1]
        winnings = np.vstack((winnings, new_row))
        trial_counter += 1

    return winnings[1:,:].T

def experiment_2(win_prob):

    winnings = realistic_simulator(1000, win_prob)
    mean = winnings.mean(axis = 1)
    median = np.median(winnings, axis = 1)
    std = winnings.std(axis = 1)

    # Figure 4
    plt.plot(mean, label='Mean')
    plt.plot(mean + std, label='Mean + Std')
    plt.plot(mean - std, label='Mean - Std')
    plt.axis([0, 300, -256, 100])
    plt.title('Figure 4')
    plt.xlabel('Roll Number')
    plt.ylabel('Cumulative Sum ($)')
    plt.legend(loc='lower right')
    plt.savefig('Figure_4.png')
    plt.close()

    # Figure 5
    plt.plot(median, label='Median')
    plt.plot(median + std, label='Median + Std')
    plt.plot(median - std, label='Median - Std')
    plt.axis([0, 300, -256, 100])
    plt.title('Figure 5')
    plt.xlabel('Roll Number')
    plt.ylabel('Cumulative Sum ($)')
    plt.legend(loc='lower right')
    plt.savefig('Figure_5.png')
    plt.close()

    # Questions
    print('Q4. ', sum(winnings[-1,:] == 80) / 1000)
    # print('Q5_+80. ', sum(winnings[-1,:] == 80))
    # print('Q5_-256. ', sum(winnings[-1,:] == -256))
    print('Q5. ', sum(winnings[-1,:]) / 1000)
    plt.title('Q6.')
    plt.plot(std, label='Standard Deviation')
    plt.xlabel('Roll Number')
    plt.legend(loc='lower right')
    plt.savefig('Q6.png')
    plt.close()

if __name__ == "__main__":
    test_code()
