#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26/04/2021

@author: Nicolò Felicioni
"""


from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Recommender_utils import similarityMatrixTopK, check_matrix
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
from sklearn.preprocessing import normalize
import numpy as np
import time
import scipy.sparse as sps

from sklearn.utils.extmath import randomized_svd

from Recommenders.Similarity.Compute_Similarity import Compute_Similarity


class EASE_Approx_2_Recommender(BaseItemSimilarityMatrixRecommender):
    """ EASE_R_Recommender

        https://arxiv.org/pdf/1905.03375.pdf

     @article{steck2019embarrassingly,
      title={Embarrassingly Shallow Autoencoders for Sparse Data},
      author={Steck, Harald},
      journal={arXiv preprint arXiv:1905.03375},
      year={2019}
    }

    """

    RECOMMENDER_NAME = "EASE_Approx_2_Recommender"


    def __init__(self, URM_train, sparse_threshold_quota = None, random_seed=42):
        super(EASE_Approx_2_Recommender, self).__init__(URM_train)
        self.sparse_threshold_quota = sparse_threshold_quota
        self.random_seed = random_seed

    def fit(self, topK=None, l2_norm = 1e3, normalize_matrix = False, n_factors=100, verbose = True):

        self.verbose = verbose

        start_time = time.time()
        self._print("Fitting model... ")

        if normalize_matrix:
            # Normalize rows and then columns
            self.URM_train = normalize(self.URM_train, norm='l2', axis=1)
            self.URM_train = normalize(self.URM_train, norm='l2', axis=0)
            self.URM_train = sps.csr_matrix(self.URM_train)


        # Woodbury formula with SVD truncated of the URM
        _, sigma, VT = randomized_svd(self.URM_train,
                                      n_components=n_factors,
                                      # n_iter=5,
                                      random_state=self.random_seed)

        # VT : k x |I|, U: |U| x k
        V = VT.T
        Z = (1/l2_norm) * VT.dot(V)
        diag_indices_n_factors = np.diag_indices(n_factors)
        Z[diag_indices_n_factors] += (1/(sigma*sigma))

        Z_inv = np.linalg.inv(Z)

        # U: n x k , Z_inv: k x k, VT: k x n ---> P: n x n
        P = (1/(l2_norm*l2_norm)) * np.dot(V, np.dot(Z_inv, VT))
        diag_indices_item_item = np.diag_indices(self.n_items)
        P[diag_indices_item_item] -= (1/l2_norm)

        P = -P

        # P = np.linalg.inv(grahm_matrix)

        B = P / (-np.diag(P))

        B[diag_indices_item_item] = 0.0


        new_time_value, new_time_unit = seconds_to_biggest_unit(time.time()-start_time)
        self._print("Fitting model... done in {:.2f} {}".format( new_time_value, new_time_unit))

        # Check if the matrix should be saved in a sparse or dense format
        # The matrix is sparse, regardless of the presence of the topK, if nonzero cells are less than sparse_threshold_quota %
        if topK is not None:
            B = similarityMatrixTopK(B, k = topK, verbose = False)


        if self._is_content_sparse_check(B):
            self._print("Detected model matrix to be sparse, changing format.")
            self.W_sparse = check_matrix(B, format='csr', dtype=np.float32)

        else:
            self.W_sparse = check_matrix(B, format='npy', dtype=np.float32)
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense
        #
        #
        # if topK is None:
        #     self.W_sparse = B
        #     self._W_sparse_format_checked = True
        #     self._compute_item_score = self._compute_score_W_dense
        #
        # else:
        #     self.W_sparse = similarityMatrixTopK(B, k = topK, verbose = False)
        #     self.W_sparse = sps.csr_matrix(self.W_sparse)


    def _is_content_sparse_check(self, matrix):

        if self.sparse_threshold_quota is None:
            return False

        if sps.issparse(matrix):
            nonzero = matrix.nnz
        else:
            nonzero = np.count_nonzero(matrix)

        return nonzero / (matrix.shape[0]**2) <= self.sparse_threshold_quota



    def _compute_score_W_dense(self, user_id_array, items_to_compute = None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        self._check_format()

        user_profile_array = self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)*np.inf
            item_scores_all = user_profile_array.dot(self.W_sparse)#.toarray()
            item_scores[:, items_to_compute] = item_scores_all[:, items_to_compute]
        else:
            item_scores = user_profile_array.dot(self.W_sparse)#.toarray()

        return item_scores





    def load_model(self, folder_path, file_name = None):
        super(EASE_Approx_2_Recommender, self).load_model(folder_path, file_name = file_name)

        if not sps.issparse(self.W_sparse):
            self._W_sparse_format_checked = True
            self._compute_item_score = self._compute_score_W_dense
