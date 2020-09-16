"""
A hybrid algorithm that incorporates input algorithms and generates a hybrid algorithm.
Different recommendation algorithms are combined based on given weights (all weights are sum up to 1).
"""
from surprise import AlgoBase

class HybridAlgoWeighted(AlgoBase):
    # store input algorithms as a dictionary: key: name of algo, value: algo object
    algorithms = {}

    # store input weights as a dictionary: key: name of algo, value: weight
    weights = {}

    weighted_estimate = 0

    
    def __init__(self, algorithms, weights):
        """

        constructor: initialize our algorithms and weights
        Args:
            param1:algorithms: the algorithms used for hybrid
            param2:weights: the weight added to the hybrid algorithm



        """
        AlgoBase.__init__(self)
        if (sum(weights.values()) - 1) != 0:
            raise Exception("Attention, sum of weights need to be 1")
       # print("in constructor, algos:", type(algorithms))
        self.algorithms = algorithms
        self.weights = weights

   
    def fit(self, trainset):
        """
        Train the model

        Args:
            param1: trainset:the training data set
        """
        AlgoBase.fit(self, trainset)

        for algoName, algo in self.algorithms.items():
            algo.fit(trainset)
        return self

    
    def estimate(self, u, i):

        """

        derived from AlgoBase: u as uid: (Raw) id of the user; i as iid: (Raw) id of the item.

        Args: 
            Param1: u : the (Raw) id of the user
            Param2:  i as iid: (Raw) id of the item.
        Return:
                weighted esimation 


        """
        #print('Hybrid Algo included (algo with weight): ')
        for algoName, algo in self.algorithms.items():
            # sum of (each algo's estimate * its weight) = weighted_estimate
            #print('', algo, ' with ', self.weights[algo])
            est = self.algorithms[algoName].estimate(u, i)
            if not isinstance(est, float):
                est = est[0]
            self.weighted_estimate += est * self.weights[algoName]

        #print('Hybrid Algo Weighted estimate: ', self.weighted_estimate)
        return self.weighted_estimate
