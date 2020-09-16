"""
The main engine for the evaluator.
"""

import csv

from DataHandler import DataHandler
from algorithm_eval import algorithm_eval
class Evaluator:
    algos = []
    movie_id_to_name = {}
    # movies = './test-data/movies.csv'
    movies = './ml-latest-small/movies.csv'



    """
    The consturctor that read the movie-to-name dataset.
    """
    def __init__(self):

        """
        The consturctor that read the movie-to-name dataset.

        """

        tempdataset = DataHandler()
        self.dataset = tempdataset
        self.readMovieName()


    def Add_Algo(self,algorithm,name):
        """
            To add the alogorithm to the evaluator

            Args:
                param1: algroithm: the added algorithm
                param2: name : the name of the algorithm

        """
        alg = algorithm_eval(algorithm,name)
        self.algos.append(alg)

    def readMovieName(self):

        """
            Read the list of movies for later use
        """
        with open(self.movies, newline='', encoding='ISO-8859-1') as csvfile:
                Reader = csv.reader(csvfile)
                next(Reader)  #Skip header line
                for row in Reader:
                    movieID = int(row[0])
                    movieName = row[1]
                    self.movie_id_to_name[movieID] = movieName


    def getMovieName(self,movieId):

        """
            Get the name by movieId

            Args:
                param1: movidId: the movieID neede to be transformed
            Return:
                the name with the movieId or an empty string
        """
        if movieId in self.movie_id_to_name.keys():
            return self.movie_id_to_name[movieId]
        return ""


    def print(self,TopN):
        """
            The print method to print the result of evaluation

            Args:
                param1: TopN: boolean whether print the TopN
        """
        result = {}
        for algo in self.algos:
            result[algo.getName()] = algo.evaluate(self.dataset,TopN)


        if(TopN):
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} ".format(
                "Algorithm", "RMSE", "MAE", "HR", "CHR","ARHR", "Diversity", "Coverage","Novelty"))
            for(name, metrics) in result.items():
                print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                      name, metrics["RMSE"], metrics["MAE"],metrics["HR"],
                      metrics["CHR"], metrics["ARHR"],
                      metrics["Diversity"], #metrics["Precision"], metrics["Recall"],
                      metrics["Coverage"],metrics["Novelty"], "Ratings HitRate"))
                print("RHR")
                print("{:<10} {:<10}".format("Ratings", "HitRate"))
                print(metrics["RHR"])
        # else:
        #     for (name, metrics) in result.items():
        #         print("{:<10} {:<10} {:<10}".format(name, metrics["RMSE"], metrics["MAE"] ))
        print("\n Note: \n")
        print("RMSE: Root Mean Squared Error")
        print("MAE: Mean Average Error")
        if(TopN):
            print("HR: Hit Rate")
            print("CHR: Cumulative Hit Rate")
            print("ARHR: Average Rank Hit Rate")
            print("Diversity: 1-Similarity")
            print("Coverage: Ratio of users for whom recommendations above a certain threshold exist.")
            print("Novelty: Average popularity rank of recommended items.")
            print("RHR: Rating Hit Rate ")



    def GenerateTopNRecs(self, testUser, N):
        """
        The method to generate top N recommendations and print the results.

        Args:
            param1: testUser : the spefic user you want to recommend for
            param2: N : the number of recommendations


        """
        for algo in self.algos:
            print("\n",algo.getName())
            #get full train data set
            print("\n Training Data:   ")

            fulltraindata = self.dataset.GetFullTrainData()

            algo.getAlgorithm().fit(fulltraindata)
            #get user anti test data set
            userAntiData = self.dataset.GetAntiUserTestData(testUser)
            #make the prediciton on unseen movies type:tuple
            predictions = algo.getAlgorithm().test(userAntiData)

            print("Top", N, "Rec - estRating")
            recommendations = []

            #since it's tuple, we have to add it into an list and then sort.
            for userID, movieID, realRating, estimatedRating,details in predictions:
                movieid_int = int(movieID)
                recommendations.append((movieid_int,estimatedRating))
            #Descending sort by estimatedRating
            recommendations.sort(key = lambda  x:x[1],reverse=True)

            # rating[0]: movieID, rating[1]: est Rating
            for rating in recommendations[:N]:
                print(self.getMovieName(rating[0]),"-",rating[1])
