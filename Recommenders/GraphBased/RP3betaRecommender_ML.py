#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 02/01/2018

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.Recommender_utils import check_matrix

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Recommender_utils import similarityMatrixTopK

from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
import os, subprocess

class RP3betaRecommender_ML(BaseItemSimilarityMatrixRecommender):
    """ RP3beta_LSQ recommender """

    #python compileCython.py RP3beta_Cython_epoch.pyx build_ext --inplace

    def __init__(self, URM_train):
        super(RP3betaRecommender_ML, self).__init__(URM_train)


    def __str__(self):
        return "RP3beta_LSQ(alpha={}, min_rating={}, topk={}, implicit={}, normalize_similarity={})".format(self.alpha,
                                                                                        self.min_rating, self.topK,
                                                                                        self.implicit, self.normalize_similarity)



    def fit(self, alpha=1., min_rating=0, topK=100, implicit=False, normalize_similarity=True,
            epochs = 30, learn_rate = 1e-2, useAdaGrad=True, objective = "RMSE"):

        self.alpha = alpha
        self.min_rating = min_rating
        self.topK = topK
        self.implicit = implicit
        self.normalize_similarity = normalize_similarity

        # Compute probabilty matrix
        p3alpha = P3alphaRecommender(self.URM_train)
        p3alpha.fit(alpha = self.alpha, min_rating=min_rating, implicit=implicit,
                    normalize_similarity=normalize_similarity, topK=topK*10)

        self.W_sparse = p3alpha.W_sparse


        from Recommenders.GraphBased.Cython.RP3beta_Cython_epoch import RP3beta_ML_Cython

        cython_model = RP3beta_ML_Cython(self.URM_train, self.W_sparse)

        itemsDegree = cython_model.fit(epochs = epochs, learn_rate = learn_rate,
                                       useAdaGrad=useAdaGrad, objective = objective)



        for item_id in range(self.URM_train.shape[1]):
            self.W_sparse[item_id,:] = self.W_sparse[item_id,:].multiply(itemsDegree)


        # CSR works faster for testing
        self.W_sparse = check_matrix(self.W_sparse, 'csr')
        self.W_sparse = similarityMatrixTopK(self.W_sparse, k=topK)

        self.URM_train = check_matrix(self.URM_train, 'csr')
