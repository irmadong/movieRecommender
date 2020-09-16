# each algorithm evaluation can take in an prediction algorithm,
# generate recommendations, and then report accuracy date
from surprise import accuracy
from collections import defaultdict

#from DataHandler import DataHandler

from DataHandler import DataHandler
import itertools as itr


class algorithm_eval:
    # constructor
    # algorithm: the prediction algorihtm we use
    # name: algorithm name
    NUM_DIGITS = 3

    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name

    """
    Getter.
    
    """
    def getName(self):
        return self.name

    def getAlgorithm(self):
        return self.algorithm

    # get mean absolute error, the lower the better. 
    def MAE(self,predictions):
        """
        
        The mean absolute error. 
        The higher the better. 


        Args:
            param1: predictions: the predicitons dataset 

        Return: 
            MAE value 
        """
        return round(accuracy.mae(predictions, verbose=False), self.NUM_DIGITS)

    # 
    def RMSE(self,predictions):

        """
        get groot mean sqrt error:
        penalize more when prediction way off, less when prediction close
        the lower the better 

        Args:
            param1: predictions: the predicitons dataset 

        """
        return round(accuracy.rmse(predictions, verbose=False), self.NUM_DIGITS)

    # return a map w/
    #     key: user, value: a list of top n estRating movies (movieID, estRating)
    def getTopN(self, predictions, n=10, ratingCutOff=4.0):
        # create a map - key: userID, value: a list of (movieID, estRating)
        res = defaultdict(list)
        res2 = defaultdict(list) # list with actualRating


        for userID, movieID, actlRating, estRating, _ in predictions:
            # if estRating is larger than rating cut-off, add the movies w/
            # estRating to the topN list of the corresponding user
            if (estRating >= ratingCutOff):
                res[int(userID)].append((int(movieID), estRating))
                res2[int(userID)].append((int(movieID), estRating, actlRating))
        # for each user-list pair, sort the list by estRating, and keep top # N
        for userID, movieList in res.items():
            sorted(movieList, key=lambda x:x[1], reverse=True)
            res[userID] = movieList[0:n] # keep top N

        for userID, movieList in res2.items():
            sorted(movieList, key=lambda x:x[1], reverse=True)
            res2[userID] = movieList[0:n] # keep top N

        return res, res2


    
    def hitRate(self,topNPred, leftOutData):


        """

        for each left out data, if the corresponding user has that movie in
        its top N list, count it as a hit. The higher the better. 

        Args:
            param1:topNPred: a dictionary w/ key: userID,
                              value: list of top N ratings (moviesID, estRating)

            param2: leftOutData: a list of left out data with high ratings from training set
        Return:
            The calculated hit rate. 


        """
        numHits = 0
        totalLeftOut = 0
        for data in leftOutData:
            userID = int(data[0])
            movieID = int(data[1])

            # check whether left-out movie is in topN list of user
            for predMovieID, _ in topNPred[userID]:
                if (movieID == predMovieID):
                    numHits += 1
                    break
            # incremental total left out data
            totalLeftOut += 1
        return round(numHits / totalLeftOut, self.NUM_DIGITS)

    
    def cumulativeHitRate(self,topNPred, leftOutData, ratingCutOff=3.0):

        """
        for each left out data, if the corresponding user has that movie in
        its top N list, count it as a hit


         Args:
            param1:topNPred: a dictionary w/ key: userID,
                                    value: list of top N ratings (moviesID, estRating)
    
            param2:leftOutData: a list of left out data with high ratings from training set

            param3:ratingCutOff: if actual rating < ratingCutOff, does not count as hits



        Return:
                 numHits / totalLeftOut, if the hits has ratings >= ratingCutOff
         
        """
        
        numHits = 0
        totalLeftOut = 0
        for data in leftOutData:
            actualRating = data[2]
            # if actual rating of left out movie >= cut off rating,
            # count hit if there exists one
            if (actualRating >= ratingCutOff):
                userID = int(data[0])
                movieID = int(data[1])

                # check whether left-out movie is in topN list of user
                for predMovieID, _ in topNPred[userID]:
                    if (movieID == predMovieID):
                        numHits += 1
                        break
                # incremental total left out data
                totalLeftOut += 1
        return round(numHits / totalLeftOut, self.NUM_DIGITS)

   
    def ratingHitRate(self,topNPred, leftOutPred):



        """
        Calculate the rating hit rate, the higher the better 
        Args:
            param1:topNPred: a dictionary w/ key: userID,
    #                            value: list of top N ratings (moviesID, estRating)
            param2:leftOutData: a list of left out data with high ratings from training set
            Returns:
             numHits / totalLeftOut foe each rating seperately
        """
        # key: rating, value: numHits / totalLeftOut corresponding to each rating
        numHits = defaultdict(float)
        totalLeftOut = defaultdict(float)

        # for each left out data, if the corresponding user has that movie in
        # its top N list, count it as a hit
        for data in leftOutPred:
            userID = int(data[0])
            movieID = int(data[1])
            actualRating = data[2]

            # check whether left-out movie is in topN list of user
            for predMovieID, _ in topNPred[userID]:
                if (movieID == predMovieID):
                    numHits[actualRating] += 1
                    break
            # incremental total left out data
            totalLeftOut[actualRating] += 1

        res = ""
        # arrange hit rates in increasing order of the corresponding ratings
        for rating in sorted(numHits.keys()):
            res += "{:<10} {:<10}\n".format(rating, round(numHits[rating]/totalLeftOut[rating], self.NUM_DIGITS))

        return res


   
    def avrgReciprocalHitRank(self, topNPred, leftOutData):


        """
        Calculate the average reciprocal hit rank the higher the better 

        Args:
            param1:a dictionary w/ key: userID,value: list of top N ratings (moviesID, estRating)
            param2:leftOutData: a list of left out data with high ratings from training set

        Return:
            rankedHits / totalLeftOut




        """



        sumRankedHits = 0
        totalLeftOut = 0
        for data in leftOutData:
            userID = int(data[0])
            movieID = int(data[1])

            # check whether left-out movie is in topN list of user
            rank = 0
            for predMovieID, _ in topNPred[userID]:
                rank += 1 # for each movie in the top N list, increment its rank
                if (movieID == predMovieID):
                    sumRankedHits += 1.0 / rank
                    break
            # incremental total left out data
            totalLeftOut += 1
            return round(sumRankedHits / totalLeftOut, self.NUM_DIGITS)

    
    
    
    def diversity(self, topNPred, simsAlgo):

        """

        Calculate how diverse the recommendation is to users by using the simsAlgo
        to compute the similarity between all pairs of recommendations for all users
        Args:
            param1:topNPred: a dictionary w/ key: userID,
    #                            value: list of top N ratings (moviesID, estRating)
            param2: simsAlgo: the algorithm used to compute similarity between movies
        Return:
            The diversity of the predictions. 





        """
        numPairs = 0
        totalSim = 0
        simsMatrix = simsAlgo.compute_similarities()
        # for each user, get all possible pairs of the user's recommended movies
        for userID in topNPred.keys():
            movieList = topNPred[userID]
            for i in range(0, len(movieList)- 1):
                for j in range(i + 1, len(movieList)):
                    # for each pair, compute their similarity
                    # first convert movieID to innerID to more easily handle data
                    innerID_i = simsAlgo.trainset.to_inner_iid(str(movieList[i][0]))
                    innerID_j = simsAlgo.trainset.to_inner_iid(str(movieList[j][0]))

                    # get similarity between movie i and j, and add sim to total sim
                    totalSim += simsMatrix[innerID_i][innerID_j]
                    numPairs += 1

        # if no recommendation is generated
        if numPairs == 0:
            diversity = -1
        else:
            similarity = totalSim / numPairs
            diversity = 1 - similarity

        return round(diversity, self.NUM_DIGITS)

  
    def userCoverage(self, topNPred, predRatingThreshold = 2.5):


        """
        Calculate the  percentage of users whose recommendations actually have
             ratings greater than or equal to the predRatingTheshold

        Args:
            param1: topNPred: a dictionary w/ key: userID,
                               value: list of top N ratings (moviesID, estRating)
            param2:numUsers: total number of user
            param3:predRatingThreshold: the threshold where if a recommended movie has ratings
                          >= the threshold, the corresponding user counts as covered
        Return: the userCoverage value




        """
        # for each user, check whether the user is covered
        numHits = 0
        numUsers = 0
        for userID in topNPred.keys():
            numUsers += 1
            # if there exists a movie in the user's recommendation whose rating
            # >= the predRatingTheshold, count as a hit
            for _, estRating in topNPred[userID]:
                if estRating >= predRatingThreshold:
                    numHits += 1
        return round(numHits / numUsers, self.NUM_DIGITS)

    # returns how new the recommended content is
    # @topNPred: a dictionary w/ key: userID,
    #                            value: list of top N ratings (moviesID, estRating)
    # @popularRankings: a dictionary with
    #                   key: movieID, value: popularity ranking of the movie (low value = high popularity)
    def novelty(self, topNPred, popularRankings):
        numMovies = 0
        totalNovelty = 0
        # for each user, get the novelty of the recommended movies
        for userID in topNPred.keys():
            for movieID, _ in topNPred[userID]:
                # add the novelty of the current movie
                totalNovelty += popularRankings[movieID]
                numMovies += 1

        # if no recommendation is generated
        if numMovies == 0:
            return -1
        return round(totalNovelty / numMovies, self.NUM_DIGITS)


    def evaluate(self, evaluationDataSet, TopN, n=10, verbose=True):
        """
        The function to generate the evaluation function. 


        Args:
            param1: evaluationDataSet: the dataset needs to be evaluated
            param2: TopN: whether we do the TopN
            param3: n: the number of top n
            param4: verbose: whether to do the print messages
        Return:
            The metrics of the evluation result.

        """
        metrics = {}
        if(verbose):
            print("Evaluating:")
        self.algorithm.fit(evaluationDataSet.GetTrainData())
        predictions_acc= self.algorithm.test(evaluationDataSet.GetTestData())

        metrics["RMSE"] = self.RMSE(predictions_acc)
        metrics["MAE"] = self.MAE(predictions_acc)
        if(TopN):
            self.algorithm.fit(evaluationDataSet.GetLOOTrain())
            #Prepare for the left one out cross validation
            looPredictions = self.algorithm.test(evaluationDataSet.GetLOOTest())
            actualPredictions = self.algorithm.test(evaluationDataSet.GetLOOAntiTestSet())
            topNPredictions, topNPredictionsWithActual = self.getTopN(actualPredictions)
            metrics["HR"] = self.hitRate(topNPredictions, looPredictions)
            metrics["CHR"] =self.cumulativeHitRate(topNPredictions,looPredictions)
            metrics["RHR"] = self.ratingHitRate(topNPredictions,looPredictions)
            metrics["ARHR"] = self.avrgReciprocalHitRank(topNPredictions,looPredictions)

            metrics["Diversity"] = self.diversity(topNPredictions, evaluationDataSet.GetSimilarities())
            metrics["Coverage"] = self.userCoverage(topNPredictions)
            metrics["Novelty"] = self.novelty(topNPredictions, evaluationDataSet.GetPopularRankings())

        # Compute accuracy
        return metrics
