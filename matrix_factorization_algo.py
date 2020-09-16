"""
This class contains all Matrix Factorization algorithms that we want to explore for our recommander system.
algorithms included:

1. SVD
2. SVDpp (SVD++): taking into account implicit ratings
3. PMF (Probabilistic Matrix Factorization): setting "biased" parameter to False
4. NMF (Non-negative Matrix Factorization)

We also want to tune our chosen algorithms (SVD & SVDpp) and here is a list of parameters we are interested:
1. n_factors: The number of factors. Default is 100.
2. n_epochs: The number of iteration of the SGD procedure. Default is 20
3. lr_all – The learning rate for all parameters. Default is 0.005.
4. reg_all – The regularization term for all parameters. Default is 0.02.

If needed, instead of tuning lr_all and reg_all we can do:
lr_bu – The learning rate for 𝑏𝑢. Takes precedence over lr_all if set. Default is None.
lr_bi – The learning rate for 𝑏𝑖. Takes precedence over lr_all if set. Default is None.
lr_pu – The learning rate for 𝑝𝑢. Takes precedence over lr_all if set. Default is None.
lr_qi – The learning rate for 𝑞𝑖. Takes precedence over lr_all if set. Default is None.
reg_bu – The regularization term for 𝑏𝑢. Takes precedence over reg_all if set. Default is None.
reg_bi – The regularization term for 𝑏𝑖. Takes precedence over reg_all if set. Default is None.
reg_pu – The regularization term for 𝑝𝑢. Takes precedence over reg_all if set. Default is None.
reg_qi – The regularization term for 𝑞𝑖. Takes precedence over reg_all if set. Default is None.

"""

from surprise import SVD, SVDpp, NMF
from surprise.model_selection import GridSearchCV


class MatrixFactorizationAlgo:
    
    def generate_algorithms(self, rating_data):
        """ here we separate untuned and tuned algo as it might take a really long time on tuning,
        it's easier to comment out the tuning part if needed
        
        Args: 
            param1: rating_data: the main data set
        Return:
                a dictionary of algorithms; key: name of algo, val: algo object

        """
        algo = {}
        algo.update({'SVD': SVD()})
        algo.update({'PMF': SVD(biased=False)})
        algo.update({'SVD++': SVDpp()})
        algo.update({'NMF': NMF()})
        print('Generated algo object for SVD, PMF, SVD++, and NMF.')

        # generate tuned SVD algorithm
        param_grid_svd = {'n_factors': [130, 200], 'n_epochs': [50, 60], 'lr_all': [0.0015, 0.002], 'reg_all' : [0.02, 0.03]}
        best_params_svd = self.tune_and_find_param('SVD', SVD, rating_data, param_grid_svd)


        # initiate tuned MF algos with tuned hyperparameters
        SVD_tuned = SVD(n_factors = best_params_svd['n_factors'],
                        n_epochs = best_params_svd['n_epochs'],
                        lr_all = best_params_svd['lr_all'])

        # append new algos to result dict
        algo.update({'SVD_tuned': SVD_tuned})

        # code for future use: tuning SVDpp, NMF
        #
        # param_grid_svdpp = {'n_factors': [20, 30], 'n_epochs': [15, 25], 'lr_all': [0.005, 0.0085]}
        # best_params_svdpp = self.tune_and_find_param('SVD++', SVDpp, rating_data, param_grid_svdpp)
        #
        # param_grid_nmf = {'n_factors': [50, 55], 'n_epochs': [45, 50], 'lr_bu': [0.02, 0.025], 'lr_bi': [0.02, 0.025]}
        # best_params_nmf = self.tune_and_find_param('NMF', NMF, rating_data, param_grid_nmf)

        # SVDpp_tuned = SVDpp(n_factors = best_params_svdpp['n_factors'],
        #                 n_epochs = best_params_svdpp['n_epochs'],
        #                 lr_all = best_params_svdpp['lr_all'])
        #
        # NMF_tuned = NMF(n_factors = best_params_nmf['n_factors'],
        #                 n_epochs = best_params_nmf['n_epochs'],
        #                 lr_bu = best_params_nmf['lr_bu'],
        #                 lr_bi = best_params_nmf['lr_bi'])
        # algo.update({'SVD++_tuned': SVDpp_tuned})
        # algo.update({'NMF_tuned': NMF_tuned})

        return algo



     
    def tune_and_find_param(self, algo_name, algo, rating_data,
        param_grid = {'n_factors': [50, 100], 'n_epochs': [20, 30], 'lr_all': [0.005, 0.010]}):
        

        """use GridSearchCVcomputes which (from surpise documentation)
        # computes accuracy metrics for an algorithm on various combinations of parameters, 
        over a cross-validation procedure.

        Args:
            param1: algo_name : the name of the algorithm
            param2: algo: the algorithm itself
            param3: rating_data: the whole dataset 

        Return:best n_factors, n_epochs, lr_all found


        """
        print("tuning for", algo_name, "hyperparameters")

        # algo: algo class name
        grid_search = GridSearchCV(algo, param_grid, measures=['rmse', 'mae'])

        # fitting data
        grid_search.fit(rating_data)

        # print the best RMSE
        print('best RMSE for ', algo_name, ' ', grid_search.best_score['rmse'])

        best_params = grid_search.best_params['rmse']
        # print the best set of parameters
        print("best params:", best_params)
        return best_params
