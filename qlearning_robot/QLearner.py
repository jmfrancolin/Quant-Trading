"""
Template for implementing QLearner  (c) 2015 Tucker Balch

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

import numpy as np
import random as rand
# import pdb

class QLearner(object):

    def __init__(self, num_states = 100, num_actions = 4, alpha = 0.2, \
        gamma = 0.9, rar = 0.5, radr = 0.99, dyna = 0, verbose = False):

        self.verbose = verbose
        self.s = 0
        self.a = 0
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        self.Q = np.zeros((num_states, num_actions))
        self.R = np.zeros((num_states, num_actions))
        self.T = np.zeros((num_states, num_actions, num_states))
        self.Tc = np.zeros((num_states, num_actions, num_states))


    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = rand.randint(0, self.num_actions - 1)
        if self.verbose: print(f"s = {s}, a = {action}")
        return action


    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        self.updateModel(self.s, self.a, s_prime, r)
        self.hallucinate()
        self.updateQ(self.s, self.a, s_prime, r)
        self.performAction(s_prime)

        if self.verbose: print(f"s = {self.s}, a = {self.a}")
        return self.a


    def performAction(self, s_prime):

        # choose a random action with probability self.rar
        if np.random.rand() < self.rar:
            self.a = rand.randint(0, self.num_actions - 1)

        # otherwise choose the action the maximizes the q-value for s_prime
        else:
            self.a = self.Q[s_prime].argmax()

        # compute random action decay
        self.rar = self.rar * self.radr

        # upate current state
        self.s = s_prime


    def updateModel(self, s, a, s_prime, r):

        # check termination condition
        if self.dyna <= 0 : return

        # keep a counter of the transitions from state s to s' given an acion a
        self.Tc[self.s, self.a, s_prime] = self.Tc[self.s, self.a, s_prime] + 1

        # T[s, a, s'] = Tc[s, a, s'] / Σ_i Tc[s, a, i]
        self.T = self.Tc / self.Tc.sum(axis = 2, keepdims = True)

        # R'[s, a] = (1- α) * R[s, a] + α * r
        self.R[self.s, self.a] = (1 - self.alpha) * self.R[self.s, self.a] + self.alpha * r


    def hallucinate(self):

        for _ in range(self.dyna):

            # pick a random state and action
            s = rand.randint(0, self.num_states - 1)
            a = rand.randint(0, self.num_actions - 1)

            # pick s_prime that maximizes q-value from (s, a) pair
            T = self.T[s, a]
            s_prime = T.argmax()

            # compute reward
            r = self.R[s, a]

            # update q-table
            self.updateQ(s, a, s_prime, r)


    def updateQ(self, s, a, s_prime, r):

        # Q'[s, a] = (1 - α) · Q[s, a] + α · (r + γ · Q[s', argmaxa'(Q[s', a'])])
        self.Q[s, a]  = (1 - self.alpha) * self.Q[s, a] + \
            self.alpha * (r + self.gamma * self.Q[s_prime, self.Q[s_prime].argmax()])


    def author(self):
        return 'jfrancolin3'


if __name__=="__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
