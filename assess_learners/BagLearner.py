import numpy as np
import pdb

class BagLearner(object):

    def __init__(self, learner, kwargs = {"leaf_size" : 1}, bags = 20, boost = False, verbose = False):

        self.learner = learner
        self.bags = bags

        self.kwargs = []
        for i in range(0, bags):
            self.kwargs.append(learner(**kwargs))

    def author(self):
        # Georgia Tech username
        return 'jfrancolin3'

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        num_rows = dataX.shape[0]
        index_candidates = np.linspace(0, num_rows - 1, num_rows)
        index_candidates = index_candidates.astype(int)

        for learner in self.kwargs:

            index = np.random.choice(index_candidates, index_candidates.size)
            learner.addEvidence(dataX.take(index, axis=0), dataY.take(index, axis=0))

    def query(self, points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        y_forecast = []

        for learner in self.kwargs:
            y_forecast = y_forecast + [learner.query(points)]
            # print(y_forecast)
            # pdb.set_trace()
            # y_forecast = np.vstack((y_forecast, learner.query(points)))

        # y_forecast_array = np.array(y_forecast)
        ans = np.mean(y_forecast_array,axis=0)

        return ans.tolist()

if __name__=="__main__":
    print("the secret clue is 'zzyzx'")
