#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/09/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Recommender_utils import check_matrix
from FW_Rating.FW_RATING_RMSE_Python import FW_RATING_RMSE_Python
import subprocess
import os, sys
import numpy as np
import numpy.matlib
import scipy.sparse as sps



class FW_RATING_RMSE_Cython(FW_RATING_RMSE_Python):


    def __init__(self, URM_train, ICM):
        super(FW_RATING_RMSE_Cython, self).__init__(URM_train, ICM)

        self.URM_train = check_matrix(URM_train, 'csr', dtype=np.float32)
        self.ICM = check_matrix(self.ICM, 'csr', dtype=np.bool)


    def initialize(self):

        # Calcolo il denominatore e poi divido element-wise
        # Se i pesi sono tutti 1 è equivalente a cosine_similarity
        self.S_denominator = numpy.matlib.repmat(self.ICM.power(2).sum(axis=1), 1, self.ICM.shape[0])
        self.S_denominator = np.sqrt(self.S_denominator)
        self.S_denominator = np.multiply(self.S_denominator, self.S_denominator.T) + self.shrink
        self.S_denominator = np.reciprocal(self.S_denominator)



        self.gradientWeights = np.ones((self.n_features))

        # Precompute gradient, which is Cosine unweighted
        self.S_weighted_gradient = self.ICM.dot(self.ICM.T).toarray()
        self.S_weighted_gradient[np.arange(self.n_items), np.arange(self.n_items)] = 0.0

        self.S_weighted_gradient = np.multiply(self.S_weighted_gradient, self.S_denominator)

        self.S_weighted = self.ICM.dot(sps.diags(self.weights)).dot(self.ICM.T).toarray()
        self.S_weighted[np.arange(self.n_items), np.arange(self.n_items)] = 0.0

        self.S_weighted = np.multiply(self.S_weighted, self.S_denominator)



    def fit(self, epochs=30, logFile=None, URM_test=None, filterTopPop = False, minRatingsPerUser=1,
            batch_size = 50, validate_every_N_epochs = 1, start_validation_after_N_epochs = 0, sgd_mode='sgd',
            lambda_w = 0.0, useBias = False, lambda_b = 0.0, learning_rate = 0.01, shrink = 100, topK = 100 ):

        self.shrink = shrink
        self.initialize()

        # Import compiled module
        from FW_Rating.Cython.FW_RATING_RMSE_Cython_Epoch import FW_RATING_RMSE_Cython_Epoch

        # Instantiate fast Cython implementation
        self.cythonEpoch = FW_RATING_RMSE_Cython_Epoch(self.ICM,
                                                 self.URM_train,
                                                 self.weights,
                                                 self.S_denominator,
                                                 self.S_weighted_gradient,
                                                 self.S_weighted,
                                                 learning_rate=learning_rate,
                                                 topK=topK,
                                                 useBias=useBias,
                                                 batch_size = batch_size,
                                                 sgd_mode=sgd_mode)

        # Cal super.fit to start training
        super(FW_RATING_RMSE_Cython, self).fit(epochs=epochs,
                                               log_file=logFile,
                                               URM_test=URM_test,
                                               filterTopPop=filterTopPop,
                                               minRatingsPerUser=minRatingsPerUser,
                                               batch_size=batch_size,
                                               validate_every_N_epochs=validate_every_N_epochs,
                                               start_validation_after_N_epochs=start_validation_after_N_epochs,
                                               sgd_mode = sgd_mode,
                                               lambda_w = lambda_w,
                                               useBias = useBias,
                                               lambda_b = lambda_b,
                                               learning_rate = learning_rate,
                                               shrink = shrink,
                                               topK = topK)




    def epochIterationBatch(self):

        #self.weights = self.cythonEpoch.epochIterationBatch_Cython()
        self.weights = self.cythonEpoch.epochIterationBoth_Cython()

        self.updateSimilarity_Linear()


    def epochIterationNoBatch(self):

        #self.weights = self.cythonEpoch.epochIterationNoBatch_Cython()
        self.weights = self.cythonEpoch.epochIterationBoth_Cython()

        self.updateSimilarity_Linear()



    def writeCurrentConfig(self, currentEpoch, results_run, logFile):

        current_config = {'lambda_w': self.lambda_w,
                          'lambda_b': self.lambda_b,
                          'use_bias': self.useBias,
                          'batch_size': self.batch_size,
                          'learn_rate': self.learning_rate,
                          'topK_similarity': self.topK,
                          'sgd_mode': self.sgd_mode,
                          'epoch': currentEpoch}

        print("Test case: {}\nResults {}\n".format(current_config, results_run))

        sys.stdout.flush()

        if (logFile != None):
            logFile.write("Test case: {}, Results {}\n".format(current_config, results_run))
            logFile.flush()

