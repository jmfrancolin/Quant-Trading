import numpy as np
# import pdb
from random import randint

class RTLearner(object):

    def __init__(self, leaf_size = 1, verbose = False):
        # constructor
        self.tree = None
        self.leaf_size = leaf_size

    def author(self):
        # Georgia Tech username
        return 'jfrancolin3'

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.tree = self.build_tree(dataX, dataY)
        print(self.tree)

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        # initialize relevant variables
        y_forecast = []
        points_row = points.shape[0]

        # loop over the matrix
        for points_row in points:

            tree_row = 0
            # transverse the tree
            while self.tree[tree_row,0] !=- 1:

                # compute parameters to follow split
                feature_index =self.tree[tree_row,0]
                split_val=self.tree[tree_row,1]

                # transverse left_tree
                if points_row[int(feature_index)] <= split_val:
                    tree_row = tree_row + int(self.tree[tree_row,2])

                # transverse right_tree
                if points_row[int(feature_index)] > split_val:
                    tree_row = tree_row + int(self.tree[tree_row,3])

            # append forecast
            y_forecast.append(self.tree[tree_row, 1])

        return y_forecast

    def build_tree(self, dataX, dataY):
        # Node Structure
        # [feature_index, split_value, reff_left_tree, reference_right_tree]
        # Node-Leaf Structure
        # [-1, Y.mean(), -1, -1]

        # check if size of data allows further splitting
        if dataX.shape[0] <= self.leaf_size:
            return np.array([-1, np.mean(dataY), -1, -1])

        # check if new information can be gainned
        if np.unique(dataY).size == 1:
            return np.array([-1, np.mean(dataY), -1, -1])

        # compute parameters to perform split
        feature_index = self.get_feature_index(dataX, dataY)
        split_val = np.median(dataX[:, feature_index])

        # construct left_tree
        mask_left_tree = dataX[:, feature_index] <= split_val
        if all(mask_left_tree):
            return np.array([-1, np.mean(dataY), -1, -1])
        left_tree = self.build_tree(dataX[mask_left_tree], dataY[mask_left_tree])

        # construct right_tree
        mask_right_tree = dataX[:, feature_index] > split_val
        if all(mask_right_tree):
            return np.array([-1, np.mean(dataY), -1, -1])
        right_tree = self.build_tree(dataX[mask_right_tree], dataY[mask_right_tree])

        # construct root
        if left_tree.ndim == 1:
            reff_righ_tree = left_tree.ndim + 1
        if left_tree.ndim > 1:
            reff_righ_tree = left_tree.shape[0] + 1
        root = np.array([feature_index, split_val, 1, reff_righ_tree])

        # concatenate subtrees and return
        return np.vstack((root, left_tree, right_tree))

    def get_feature_index(self, dataX, dataY):
        # retrun a random integer in the range [0, num_coll)
        return randint(0, dataX.shape[1] - 1)

if __name__=="__main__":
    print("the secret clue is 'zzyzx'")
