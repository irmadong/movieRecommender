
from Evaluator import Evaluator
from surprise import NormalPredictor
from matrix_factorization_algo import MatrixFactorizationAlgo
from DataHandler import DataHandler

from knn_algo import knn
from hybrid_algo_weighted import HybridAlgoWeighted

# from Evaluator import Evaluator
#load dataset
dataprocessor = DataHandler()
evaluationData = dataprocessor.getEvaluation()
rankings = dataprocessor.getRank()
evaluator = Evaluator()

#use random as our basline here
Random = NormalPredictor()
evaluator.Add_Algo(Random, "Random")

# add knn algos
knngenerator = knn()
knn_algo_dict = knngenerator.generate_knn(evaluationData)
for key in knn_algo_dict:
    evaluator.Add_Algo(knn_algo_dict[key],key)

# adding MF algos
mf_algo = MatrixFactorizationAlgo()
mf_algo_dict = mf_algo.generate_algorithms(evaluationData)
for key in mf_algo_dict:
    evaluator.Add_Algo(mf_algo_dict[key], key)

# since KNN_bl has high novelty and diversity, SVD++ has lower novelty and diversity,
# we can combine the KNN and SVD approaches in a way based on how much a user likes
# novelty & diversity

# Suppose a user has been enjoying novel and diverse recommendations, the
# user's preference towards novelty and diversity is high, so we will assign more
# weight to the KNN which yields more novelty and diversity;
# on the other hand if a user dislikes novel and diverse recommendations,
# we will assign more weight to the SVD algorithm, according to that user's
# preference in the hybrid approach
#
# here is an illustration of the hybrid approach we are describing. Assume the
# user preference towards diversity and novelty is 0.8
userPref_novel_diverse = 0.8  # could be computed and changed in the future

hybrid_weighted_algorithms = {'blKNN_tuned' : knn_algo_dict['blKNN_tuned'], 'SVD_tuned' : mf_algo_dict['SVD_tuned']}

# hybrid 1 with high novelty and diversity
hybrid_weighted_weights = {'blKNN_tuned' : userPref_novel_diverse, 'SVD_tuned' : 1 - userPref_novel_diverse}
hybrid_weighted = HybridAlgoWeighted(hybrid_weighted_algorithms, hybrid_weighted_weights)
evaluator.Add_Algo(hybrid_weighted, "hybrid")


# evaluate
evaluator.print(True)

# print recommendations for user
dummyUserID = 11
N = 5
evaluator.GenerateTopNRecs(dummyUserID, N)
