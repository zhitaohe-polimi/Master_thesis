#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from Recommenders.Recommender_utils import check_matrix

import sys
import numpy as np


class new_algo_with_MFAttetion_Cython(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):
    RECOMMENDER_NAME = "MatrixFactorization_Cython_Recommender"

    def __init__(self, URM_train, verbose=True, algorithm_name="FUNK_SVD"):
        super(new_algo_with_MFAttetion_Cython, self).__init__(URM_train, verbose=verbose)

        self.n_users, self.n_items = self.URM_train.shape
        self.normalize = False
        self.algorithm_name = algorithm_name

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        USER_factors is n_users x n_factors
        ITEM_factors is n_items x n_factors

        The prediction for cold users will always be -inf for ALL items

        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        assert self.USER_factors.shape[1] == self.ITEM_factors.shape[1], \
            "{}: User and Item factors have inconsistent shape".format(self.RECOMMENDER_NAME)

        assert self.USER_factors.shape[0] > np.max(user_id_array), \
            "{}: Cold users not allowed. Users in trained model are {}, requested prediction for users up to {}".format(
                self.RECOMMENDER_NAME, self.USER_factors.shape[0], np.max(user_id_array))

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.ITEM_factors.shape[0]), dtype=np.float32) * np.inf
            item_scores[:, items_to_compute] = np.dot(self.USER_factors[user_id_array],
                                                      self.ITEM_factors[items_to_compute, :].T) \
                                               + np.dot(self.similarity_matrix_user[user_id_array],
                                                        np.dot(self.USER_factors_user[user_id_array],
                                                               self.ITEM_factors_user[items_to_compute, :].T)) \
                                               + np.dot(self.similarity_matrix_item[items_to_compute],
                                                        np.dot(self.USER_factors_item[user_id_array],
                                                               self.ITEM_factors_item[items_to_compute, :].T))

        else:
            item_scores = np.dot(self.USER_factors[user_id_array], self.ITEM_factors.T) \
                          + np.dot(self.similarity_matrix_user[user_id_array],
                                   np.dot(self.USER_factors_user[user_id_array], self.ITEM_factors_user.T)) \
                          + np.dot(self.similarity_matrix_item,
                                   np.dot(self.USER_factors_item[user_id_array], self.ITEM_factors_item.T))

        # No need to select only the specific negative items or warm users because the -inf score will not change
        if self.use_bias:
            item_scores += self.ITEM_bias + self.GLOBAL_bias
            item_scores = (item_scores.T + self.USER_bias[user_id_array]).T

        return item_scores

    def fit(self, epochs=300, batch_size=1000,
            num_factors=10,
            num_factors_user=10,
            num_factors_item=10,
            positive_threshold_BPR=None,
            learning_rate=0.001,
            learning_rate_user=0.001,
            learning_rate_item=0.001,
            use_bias=True,
            use_embeddings=True,
            sgd_mode='sgd',
            negative_interactions_quota=0.0,
            dropout_quota=None,
            init_mean=0.0, init_std_dev=0.1,
            user_reg=0.0, item_reg=0.0, bias_reg=0.0, positive_reg=0.0, negative_reg=0.0,
            user_reg_u=0.0, item_reg_u=0.0, user_reg_i=0.0, item_reg_i=0.0,
            random_seed=None,
            **earlystopping_kwargs):

        self.num_factors = num_factors
        self.num_factors_user = num_factors_user
        self.num_factors_item = num_factors_item
        self.use_bias = use_bias
        self.sgd_mode = sgd_mode
        self.positive_threshold_BPR = positive_threshold_BPR
        self.learning_rate = learning_rate
        self.learning_rate_user = learning_rate_user
        self.learning_rate_item = learning_rate_item

        assert negative_interactions_quota >= 0.0 and negative_interactions_quota < 1.0, "{}: negative_interactions_quota must be a float value >=0 and < 1.0, provided was '{}'".format(
            self.RECOMMENDER_NAME, negative_interactions_quota)
        self.negative_interactions_quota = negative_interactions_quota

        # Import compiled module
        from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython_Epoch import \
            MatrixFactorization_Cython_Epoch

        if self.algorithm_name in ["FUNK_SVD"]:
            self.cythonEpoch = MatrixFactorization_Cython_Epoch(self.URM_train,
                                                                algorithm_name=self.algorithm_name,
                                                                n_factors=self.num_factors,
                                                                n_factors_user=self.num_factors_user,
                                                                n_factors_item=self.num_factors_item,
                                                                learning_rate=learning_rate,
                                                                learning_rate_user=learning_rate_user,
                                                                learning_rate_item=learning_rate_item,
                                                                sgd_mode=sgd_mode,
                                                                user_reg=user_reg,
                                                                item_reg=item_reg,
                                                                user_reg_u=user_reg_u,
                                                                item_reg_u=item_reg_u,
                                                                user_reg_i=user_reg_i,
                                                                item_reg_i=item_reg_i,
                                                                bias_reg=bias_reg,
                                                                batch_size=batch_size,
                                                                use_bias=use_bias,
                                                                use_embeddings=use_embeddings,
                                                                init_mean=init_mean,
                                                                negative_interactions_quota=negative_interactions_quota,
                                                                dropout_quota=dropout_quota,
                                                                init_std_dev=init_std_dev,
                                                                verbose=self.verbose,
                                                                random_seed=random_seed)

        self._prepare_model_for_validation()
        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.algorithm_name,
                                        **earlystopping_kwargs)

        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best
        self.USER_factors_user = self.USER_factors_user_best
        self.ITEM_factors_user = self.ITEM_factors_user_best
        self.USER_factors_item = self.USER_factors_item_best
        self.ITEM_factors_item = self.ITEM_factors_item_best

        if self.use_bias:
            self.USER_bias = self.USER_bias_best
            self.ITEM_bias = self.ITEM_bias_best
            self.GLOBAL_bias = self.GLOBAL_bias_best

        sys.stdout.flush()

    def _prepare_model_for_validation(self):
        self.USER_factors = self.cythonEpoch.get_USER_factors()
        self.ITEM_factors = self.cythonEpoch.get_ITEM_factors()
        self.USER_factors_user = self.cythonEpoch.get_USER_factors_user()
        self.ITEM_factors_user = self.cythonEpoch.get_ITEM_factors_user()
        self.USER_factors_item = self.cythonEpoch.get_USER_factors_item()
        self.ITEM_factors_item = self.cythonEpoch.get_ITEM_factors_item()
        self.similarity_matrix_user = self.cythonEpoch.get_similarity_matrix_user()
        self.similarity_matrix_item = self.cythonEpoch.get_similarity_matrix_item()

        if self.use_bias:
            self.USER_bias = self.cythonEpoch.get_USER_bias()
            self.ITEM_bias = self.cythonEpoch.get_ITEM_bias()
            self.GLOBAL_bias = self.cythonEpoch.get_GLOBAL_bias()

    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()
        self.USER_factors_user_best = self.USER_factors_user.copy()
        self.ITEM_factors_user_best = self.ITEM_factors_user.copy()
        self.USER_factors_item_best = self.USER_factors_item.copy()
        self.ITEM_factors_item_best = self.ITEM_factors_item.copy()

        if self.use_bias:
            self.USER_bias_best = self.USER_bias.copy()
            self.ITEM_bias_best = self.ITEM_bias.copy()
            self.GLOBAL_bias_best = self.GLOBAL_bias

    def _run_epoch(self, num_epoch):
        self.cythonEpoch.epochIteration_Cython()


class new_MatrixFactorization_FunkSVD_Cython(new_algo_with_MFAttetion_Cython):
    """
    Subclas allowing only for FunkSVD model

    Reference: http://sifter.org/~simon/journal/20061211.html

    Factorizes the rating matrix R into the dot product of two matrices U and V of latent factors.
    U represent the user latent factors, V the item latent factors.
    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin} \limits_{U,V}\frac{1}{2}||R - UV^T||^2_2 + \frac{\lambda}{2}(||U||^2_F + ||V||^2_F)
    Latent factors are initialized from a Normal distribution with given mean and std.

    """

    RECOMMENDER_NAME = "new_MatrixFactorization_FunkSVD_Cython_Recommender"

    def __init__(self, *pos_args, **key_args):
        super(new_MatrixFactorization_FunkSVD_Cython, self).__init__(*pos_args, algorithm_name="FUNK_SVD", **key_args)

    def fit(self, **key_args):
        super(new_MatrixFactorization_FunkSVD_Cython, self).fit(**key_args)
