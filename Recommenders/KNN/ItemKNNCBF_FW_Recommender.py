#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/03/19

@author: Simone Boglio
"""

from Recommenders.Recommender_utils import check_matrix
from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender
from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np
import scipy.sparse as sps

from Recommenders.Similarity.Compute_Similarity import Compute_Similarity


class ItemKNNCBF_FW_Recommender(BaseItemCBFRecommender, BaseItemSimilarityMatrixRecommender):
    """ ItemKNN recommender"""

    RECOMMENDER_NAME = "ItemKNNCBF_FW_Recommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, ICM_list, URM_train):

        assert False, "rotto"
        super(ItemKNNCBF_FW_Recommender, self).__init__(URM_train)

        self.n_icm = len(ICM_list)

        self.ICM_train_list = []
        for ICM in ICM_list:
            self.ICM_train_list.append(ICM.copy())



    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting = "none", **similarity_args):
        self.ICM_train_weigth_list = []
        key='icm_wX'
        try:
            for i in range (self.n_icm):
                key = 'icm_w{}'.format(int(i))
                self.ICM_train_weigth_list.append(float(similarity_args[key]))
                similarity_args.pop(key, None)
        except:
            print('{}: No weight for {} in fit function.'.format(self.RECOMMENDER_NAME, key))

        assert len(self.ICM_train_weigth_list) == len(self.ICM_train_list), '{}: number weights and number matrixes not equal'.format(self.RECOMMENDER_NAME)

        for i in range (self.n_icm):
            self.ICM_train_list[i] = self.ICM_train_list[i] * self.ICM_train_weigth_list[i]
            self.ICM_train_list[i].eliminate_zeros()

        self.ICM_train = sps.hstack(self.ICM_train_list)

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError("Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(self.FEATURE_WEIGHTING_VALUES, feature_weighting))


        if feature_weighting == "BM25":
            self.ICM_train = self.ICM_train.astype(np.float32)
            self.ICM_train = okapi_BM_25(self.ICM_train)

        elif feature_weighting == "TF-IDF":
            self.ICM_train = self.ICM_train.astype(np.float32)
            self.ICM_train = TF_IDF(self.ICM_train)


        similarity = Compute_Similarity(self.ICM_train.T, shrink=shrink, topK=topK, normalize=normalize, similarity = similarity, **similarity_args)


        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')


