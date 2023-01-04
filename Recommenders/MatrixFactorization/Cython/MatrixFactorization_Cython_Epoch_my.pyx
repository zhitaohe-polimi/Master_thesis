"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""

#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from Recommenders.Recommender_utils import check_matrix
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit

import cython

import numpy as np
cimport numpy as np
import time, math
import sys

from libc.math cimport exp, sqrt
from libc.stdlib cimport rand, srand, RAND_MAX
from sklearn.preprocessing import normalize


cdef struct BPR_sample:
    long user
    long pos_item
    long neg_item


cdef struct MSE_sample:
    long user
    long item
    double rating



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
cdef class MatrixFactorization_Cython_Epoch:

    cdef int n_users, n_items, n_factors, n_factors_user, n_factors_item,print_step_seconds
    cdef algorithm_name

    cdef double learning_rate, learning_rate_user, learning_rate_item, user_reg, item_reg, user_reg_u, item_reg_u, user_reg_i, item_reg_i, positive_reg, negative_reg, bias_reg
    cdef double init_mean, init_std_dev, MSE_negative_interactions_quota, MSE_sample_negative_interactions_flag

    cdef int batch_size

    cdef int algorithm_is_funk_svd, algorithm_is_asy_svd, algorithm_is_BPR

    cdef int[:] URM_train_indices, URM_train_indptr, profile_length
    cdef double[:] URM_train_data

    cdef double[:,:] USER_factors, ITEM_factors, USER_factors_user, ITEM_factors_user, USER_factors_item, ITEM_factors_item
    cdef double[:] USER_bias, ITEM_bias, GLOBAL_bias
    cdef int[:] factors_dropout_mask,factors_dropout_mask_user,factors_dropout_mask_item
    cdef int dropout_flag
    cdef double dropout_quota


    # Mini-batch sample data
    cdef double[:,:] USER_factors_minibatch_accumulator, ITEM_factors_minibatch_accumulator, USER_factors_minibatch_accumulator_user, ITEM_factors_minibatch_accumulator_user, USER_factors_minibatch_accumulator_item, ITEM_factors_minibatch_accumulator_item
    cdef double[:] USER_bias_minibatch_accumulator, ITEM_bias_minibatch_accumulator, GLOBAL_bias_minibatch_accumulator

    cdef long[:] mini_batch_sampled_items, mini_batch_sampled_users
    cdef long[:] mini_batch_sampled_items_flag, mini_batch_sampled_users_flag
    cdef long mini_batch_sampled_items_counter, mini_batch_sampled_users_counter

    #

    cdef int useAdaGrad, useRmsprop, useAdam, verbose, use_bias, use_embeddings

    cdef double [:,:] sgd_cache_I, sgd_cache_U, sgd_cache_I_user, sgd_cache_U_user, sgd_cache_I_item, sgd_cache_U_item, sgd_cache_bias_I, sgd_cache_bias_U, sgd_cache_bias_GLOBAL
    cdef double gamma

    cdef double [:,:] sgd_cache_I_momentum_1, sgd_cache_I_momentum_2, sgd_cache_I_momentum_1_user, sgd_cache_I_momentum_2_user, sgd_cache_I_momentum_1_item, sgd_cache_I_momentum_2_item
    cdef double [:,:] sgd_cache_U_momentum_1, sgd_cache_U_momentum_2, sgd_cache_U_momentum_1_user, sgd_cache_U_momentum_2_user, sgd_cache_U_momentum_1_item, sgd_cache_U_momentum_2_item
    cdef double [:,:] sgd_cache_bias_I_momentum_1, sgd_cache_bias_I_momentum_2
    cdef double [:,:] sgd_cache_bias_U_momentum_1, sgd_cache_bias_U_momentum_2
    cdef double [:,:] sgd_cache_bias_GLOBAL_momentum_1, sgd_cache_bias_GLOBAL_momentum_2
    cdef  similarity_matrix_user, similarity_matrix_item
    cdef double beta_1, beta_2, beta_1_power_t, beta_2_power_t
    cdef double momentum_1, momentum_2

    SGD_MODE_VALUES = ["sgd", "adam", "adagrad", "rmsprop"]
    ALGORITHM_NAME_VALUES = ["FUNK_SVD", "ASY_SVD", "MF_BPR"]


    def __init__(self, URM_train, n_factors = 1, n_factors_user = 1, n_factors_item = 1,
                 algorithm_name = None,
                 batch_size = 1,
                 negative_interactions_quota = 0.5,
                 dropout_quota = None,
                 learning_rate = 1e-3, learning_rate_user = 1e-3, learning_rate_item = 1e-3,
                 use_bias = False, use_embeddings = True,
                 user_reg = 0.0, item_reg = 0.0, bias_reg = 0.0, positive_reg = 0.0, negative_reg = 0.0,
                 user_reg_u = 0.0, item_reg_u = 0.0, user_reg_i = 0.0, item_reg_i=0.0,
                 verbose = False, print_step_seconds = 300, random_seed = None,
                 init_mean = 0.0, init_std_dev = 0.1,
                 sgd_mode='sgd', gamma=0.995, beta_1=0.9, beta_2=0.999):

        super(MatrixFactorization_Cython_Epoch, self).__init__()


        if sgd_mode not in self.SGD_MODE_VALUES:
           raise ValueError("Value for 'sgd_mode' not recognized. Acceptable values are {}, provided was '{}'".format(self.SGD_MODE_VALUES, sgd_mode))

        if algorithm_name not in self.ALGORITHM_NAME_VALUES:
           raise ValueError("Value for 'algorithm_name' not recognized. Acceptable values are {}, provided was '{}'".format(self.ALGORITHM_NAME_VALUES, algorithm_name))


        # Create copy of URM_train in csr format
        # make sure indices are sorted
        URM_train = check_matrix(URM_train, 'csr')
        URM_train = URM_train.sorted_indices()

        self.profile_length = np.ediff1d(URM_train.indptr)
        self.n_users, self.n_items = URM_train.shape

        #compute similarity between users or items
        # URM_train_array=URM_train.toarray()
        # print("URM shape: ",URM_train_array.shape)
        # self.similarity_matrix_user = URM_train_array.dot(URM_train_array.T)
        # self.similarity_matrix_user=self.similarity_matrix_user.toarray()
        # print("similarity_matrix_user ",self.similarity_matrix_user.shape)
        # self.similarity_matrix_item = URM_train_array.T.dot(URM_train_array)
        # self.similarity_matrix_item=self.similarity_matrix_item.toarray()
        # print("similarity_matrix_item ",self.similarity_matrix_item.shape)

        self.similarity_matrix_user = normalize(URM_train @ URM_train.transpose(),norm='l2',axis=1).toarray()
        print("a ",self.similarity_matrix_user.shape)

        self.similarity_matrix_item = normalize(URM_train.transpose() @ URM_train,norm='l2',axis=1).toarray()
        print("b ",self.similarity_matrix_item.shape)


        self.n_factors = n_factors
        self.n_factors_user = n_factors_user
        self.n_factors_item = n_factors_item
        self.verbose = verbose
        self.algorithm_name = algorithm_name
        self.learning_rate = learning_rate
        self.learning_rate_user = learning_rate_user
        self.learning_rate_item = learning_rate_item
        self.user_reg = user_reg
        self.item_reg = item_reg
        self.user_reg_u = user_reg_u
        self.item_reg_u = item_reg_u
        self.user_reg_i = user_reg_i
        self.item_reg_i = item_reg_i
        self.positive_reg = positive_reg
        self.negative_reg = negative_reg
        self.bias_reg = bias_reg
        self.print_step_seconds = print_step_seconds

        assert (use_bias or use_embeddings), "At least one between use_bias and use_embeddings must be True"
        self.use_bias = use_bias
        self.use_embeddings = use_embeddings

        self.batch_size = batch_size
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.MSE_negative_interactions_quota = negative_interactions_quota
        self.MSE_sample_negative_interactions_flag = self.MSE_negative_interactions_quota != 0.0

        self.URM_train_indices = URM_train.indices
        self.URM_train_data = np.array(URM_train.data, dtype=np.float64)
        self.URM_train_indptr = URM_train.indptr

        if random_seed is not None:
            np.random.seed(seed=random_seed)
            srand(<unsigned int> int(random_seed))


        self.factors_dropout_mask = np.ones(self.n_factors, dtype=np.int32)
        self.factors_dropout_mask_user = np.ones(self.n_factors_user, dtype=np.int32)
        self.factors_dropout_mask_item = np.ones(self.n_factors_item, dtype=np.int32)

        if dropout_quota is None:
            self.dropout_flag = False
            self.dropout_quota = 0.0
        else:
            assert dropout_quota>0.0 and dropout_quota<1.0, "Value for 'dropout_quota' not valid. Acceptable values are None or a float value >0.0 and <1.0, provided value was '{}'".format(dropout_quota)
            self.dropout_flag = True
            self.dropout_quota = dropout_quota


        self._init_latent_factors()
        self._init_minibatch_data_structures()
        self._init_adaptive_gradient_cache(sgd_mode, gamma, beta_1, beta_2)



    def _init_latent_factors(self):

        self.algorithm_is_funk_svd = False
        self.algorithm_is_asy_svd = False
        self.algorithm_is_BPR = False

        n_user_factors = self.n_users
        n_item_factors = self.n_items

        if self.algorithm_name == "FUNK_SVD":
            self.algorithm_is_funk_svd = True

        # elif self.algorithm_name == "ASY_SVD":
        #     self.algorithm_is_asy_svd = True
        #     n_user_factors = self.n_items
        #     n_item_factors = self.n_items
        #
        # elif self.algorithm_name == "MF_BPR":
        #     assert self.use_embeddings, "For MF_BPR use_embeddings must be True"
        #     self.algorithm_is_BPR = True


        if self.use_embeddings:
            # W and H cannot be initialized as zero, otherwise the gradient will always be zero
            self.USER_factors = np.random.normal(self.init_mean, self.init_std_dev, (n_user_factors, self.n_factors)).astype(np.float64)
            self.ITEM_factors = np.random.normal(self.init_mean, self.init_std_dev, (n_item_factors, self.n_factors)).astype(np.float64)

            self.USER_factors_minibatch_accumulator = np.zeros((n_user_factors, self.n_factors), dtype=np.float64)
            self.ITEM_factors_minibatch_accumulator = np.zeros((n_item_factors, self.n_factors), dtype=np.float64)

            #W and H for User attention
            self.USER_factors_user = np.random.normal(self.init_mean, self.init_std_dev,
                                                 (n_user_factors, self.n_factors_user)).astype(np.float64)
            self.ITEM_factors_user = np.random.normal(self.init_mean, self.init_std_dev,
                                                 (n_item_factors, self.n_factors_user)).astype(np.float64)

            self.USER_factors_minibatch_accumulator_user = np.zeros((n_user_factors, self.n_factors_user), dtype=np.float64)
            self.ITEM_factors_minibatch_accumulator_user = np.zeros((n_item_factors, self.n_factors_user), dtype=np.float64)

            #W and H for item attention
            self.USER_factors_item = np.random.normal(self.init_mean, self.init_std_dev,
                                                 (n_user_factors, self.n_factors_item)).astype(np.float64)
            self.ITEM_factors_item = np.random.normal(self.init_mean, self.init_std_dev,
                                                 (n_item_factors, self.n_factors_item)).astype(np.float64)

            self.USER_factors_minibatch_accumulator_item = np.zeros((n_user_factors, self.n_factors_item), dtype=np.float64)
            self.ITEM_factors_minibatch_accumulator_item = np.zeros((n_item_factors, self.n_factors_item), dtype=np.float64)
        else:
            self.USER_factors = np.zeros((n_user_factors, self.n_factors_item)).astype(np.float64)
            self.ITEM_factors = np.zeros((n_item_factors, self.n_factors_item)).astype(np.float64)

            #user_embeddings is always set as True



        if self.use_bias:
            self.USER_bias = np.zeros(self.n_users, dtype=np.float64)
            self.ITEM_bias = np.zeros(self.n_items, dtype=np.float64)
            self.GLOBAL_bias = np.zeros(1, dtype=np.float64)

            self.USER_bias_minibatch_accumulator = np.zeros(self.n_users, dtype=np.float64)
            self.ITEM_bias_minibatch_accumulator = np.zeros(self.n_items, dtype=np.float64)
            self.GLOBAL_bias_minibatch_accumulator = np.zeros(1, dtype=np.float64)





    def _init_adaptive_gradient_cache(self, sgd_mode, gamma, beta_1, beta_2):

        self.useAdaGrad = False
        self.useRmsprop = False
        self.useAdam = False

        if sgd_mode=='adagrad':
            self.useAdaGrad = True

        elif sgd_mode=='rmsprop':
            self.useRmsprop = True

            # Gamma default value suggested by Hinton
            # self.gamma = 0.9
            self.gamma = gamma

        elif sgd_mode=='adam':
            self.useAdam = True

            # Default value suggested by the original paper
            # beta_1=0.9, beta_2=0.999
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.beta_1_power_t = beta_1
            self.beta_2_power_t = beta_2



        if sgd_mode=='sgd':
            self.sgd_cache_I = None
            self.sgd_cache_U = None

            self.sgd_cache_I_user = None
            self.sgd_cache_U_user = None

            self.sgd_cache_I_item = None
            self.sgd_cache_U_item = None

            self.sgd_cache_bias_I = None
            self.sgd_cache_bias_U = None
            self.sgd_cache_bias_GLOBAL = None

            self.sgd_cache_I_momentum_1 = None
            self.sgd_cache_I_momentum_2 = None

            self.sgd_cache_I_momentum_1_user = None
            self.sgd_cache_I_momentum_2_user = None

            self.sgd_cache_I_momentum_1_item = None
            self.sgd_cache_I_momentum_2_item = None

            self.sgd_cache_U_momentum_1 = None
            self.sgd_cache_U_momentum_2 = None

            self.sgd_cache_U_momentum_1_user = None
            self.sgd_cache_U_momentum_2_user = None

            self.sgd_cache_U_momentum_1_item = None
            self.sgd_cache_U_momentum_2_item = None

            self.sgd_cache_bias_I_momentum_1 = None
            self.sgd_cache_bias_I_momentum_2 = None

            self.sgd_cache_bias_U_momentum_1 = None
            self.sgd_cache_bias_U_momentum_2 = None

            self.sgd_cache_bias_GLOBAL_momentum_1 = None
            self.sgd_cache_bias_GLOBAL_momentum_2 = None

        else:

            # Adagrad and RMSProp
            self.sgd_cache_I = np.zeros((self.ITEM_factors.shape[0], self.n_factors), dtype=np.float64)
            self.sgd_cache_U = np.zeros((self.USER_factors.shape[0], self.n_factors), dtype=np.float64)

            self.sgd_cache_I_user = np.zeros((self.ITEM_factors_user.shape[0], self.n_factors_user), dtype=np.float64)
            self.sgd_cache_U_user = np.zeros((self.USER_factors_user.shape[0], self.n_factors_user), dtype=np.float64)

            self.sgd_cache_I_item = np.zeros((self.ITEM_factors_item.shape[0], self.n_factors_item), dtype=np.float64)
            self.sgd_cache_U_item = np.zeros((self.USER_factors_item.shape[0], self.n_factors_item), dtype=np.float64)

            self.sgd_cache_bias_I = np.zeros((self.n_items, 1), dtype=np.float64)
            self.sgd_cache_bias_U = np.zeros((self.n_users, 1), dtype=np.float64)
            self.sgd_cache_bias_GLOBAL = np.zeros((1, 1), dtype=np.float64)

            # Adam
            self.sgd_cache_I_momentum_1 = np.zeros((self.ITEM_factors.shape[0], self.n_factors), dtype=np.float64)
            self.sgd_cache_I_momentum_2 = np.zeros((self.ITEM_factors.shape[0], self.n_factors), dtype=np.float64)

            self.sgd_cache_I_momentum_1_user = np.zeros((self.ITEM_factors_user.shape[0], self.n_factors_user), dtype=np.float64)
            self.sgd_cache_I_momentum_2_user = np.zeros((self.ITEM_factors_user.shape[0], self.n_factors_user), dtype=np.float64)

            self.sgd_cache_I_momentum_1_item = np.zeros((self.ITEM_factors_item.shape[0], self.n_factors_item), dtype=np.float64)
            self.sgd_cache_I_momentum_2_item = np.zeros((self.ITEM_factors_item.shape[0], self.n_factors_item), dtype=np.float64)

            self.sgd_cache_U_momentum_1 = np.zeros((self.USER_factors.shape[0], self.n_factors), dtype=np.float64)
            self.sgd_cache_U_momentum_2 = np.zeros((self.USER_factors.shape[0], self.n_factors), dtype=np.float64)

            self.sgd_cache_U_momentum_1_user = np.zeros((self.USER_factors_user.shape[0], self.n_factors_user), dtype=np.float64)
            self.sgd_cache_U_momentum_2_user = np.zeros((self.USER_factors_user.shape[0], self.n_factors_user), dtype=np.float64)

            self.sgd_cache_U_momentum_1_item = np.zeros((self.USER_factors_item.shape[0], self.n_factors_item), dtype=np.float64)
            self.sgd_cache_U_momentum_2_item = np.zeros((self.USER_factors_item.shape[0], self.n_factors_item), dtype=np.float64)

            self.sgd_cache_bias_I_momentum_1 = np.zeros((self.n_items, 1), dtype=np.float64)
            self.sgd_cache_bias_I_momentum_2 = np.zeros((self.n_items, 1), dtype=np.float64)

            self.sgd_cache_bias_U_momentum_1 = np.zeros((self.n_users, 1), dtype=np.float64)
            self.sgd_cache_bias_U_momentum_2 = np.zeros((self.n_users, 1), dtype=np.float64)

            self.sgd_cache_bias_GLOBAL_momentum_1 = np.zeros((1, 1), dtype=np.float64)
            self.sgd_cache_bias_GLOBAL_momentum_2 = np.zeros((1, 1), dtype=np.float64)



    def epochIteration_Cython(self):

        if self.algorithm_is_funk_svd:
            self.epochIteration_Cython_FUNK_SVD_SGD()



    def epochIteration_Cython_FUNK_SVD_SGD(self):

        # Get number of available interactions
        cdef long n_total_batch = int(len(self.URM_train_data) / self.batch_size) + 1

        cdef MSE_sample sample
        cdef long factor_index, n_current_batch, n_sample_in_batch, processed_samples_last_print=0, print_block_size = 500
        cdef double prediction, prediction_error
        cdef double local_gradient_item, local_gradient_user, local_gradient_bias_item, local_gradient_bias_user, local_gradient_bias_global

        cdef double H_i, W_u, cumulative_loss = 0.0


        cdef long start_time_epoch = time.time()
        cdef long last_print_time = start_time_epoch

        # Renew dropout mask
        if self.dropout_flag:
            for factor_index in range(self.n_factors):
                self.factors_dropout_mask[factor_index] = rand() > self.dropout_quota

            for factor_index in range(self.n_factors_user):
                self.factors_dropout_mask_user[factor_index] = rand() > self.dropout_quota

            for factor_index in range(self.n_factors_item):
                self.factors_dropout_mask_item[factor_index] = rand() > self.dropout_quota

            if self.n_factors == 1:
                self.factors_dropout_mask[0] = True

            if self.n_factors_user == 1:
                self.factors_dropout_mask_user[0] = True

            if self.n_factors_item == 1:
                self.factors_dropout_mask_item[0] = True

        #item_scores = np.dot(self.USER_factors, self.ITEM_factors.T)
        item_scores_for_user = np.dot(self.USER_factors_user, self.ITEM_factors_user.T)
        item_scores_for_item = np.dot(self.USER_factors_item, self.ITEM_factors_item.T)

        for n_current_batch in range(n_total_batch):

            self._clear_minibatch_data_structures()

            # Iterate over samples in batch
            for n_sample_in_batch in range(self.batch_size):

                # Uniform user sampling with replacement
                sample = self.sampleMSE_Cython()

                self._add_MSE_sample_in_minibatch(sample)

                # Compute prediction
                if self.use_bias:
                    prediction = self.GLOBAL_bias[0] + self.USER_bias[sample.user] + self.ITEM_bias[sample.item]
                else:
                    prediction = 0.0

                if self.use_embeddings:
                    # item_scores=np.dot(self.USER_factors,self.ITEM_factors.T)
                    for factor_index in range(self.n_factors):
                        if self.factors_dropout_mask[factor_index]:
                            prediction += self.USER_factors[sample.user, factor_index] * self.ITEM_factors[sample.item, factor_index]

                    prediction += np.dot( self.similarity_matrix_user[sample.user,:] , item_scores_for_user[:, sample.item] )
                    prediction += np.dot( self.similarity_matrix_item[sample.item,:] , item_scores_for_item.T[:, sample.user] )


                # Compute gradients
                prediction_error = sample.rating - prediction
                cumulative_loss += prediction_error**2


                if self.use_bias:
                    local_gradient_bias_global = prediction_error - self.bias_reg * self.GLOBAL_bias[0]
                    local_gradient_bias_item = prediction_error - self.bias_reg * self.ITEM_bias[sample.item]
                    local_gradient_bias_user = prediction_error - self.bias_reg * self.USER_bias[sample.user]

                    self.GLOBAL_bias_minibatch_accumulator[0] += local_gradient_bias_global
                    self.ITEM_bias_minibatch_accumulator[sample.item] += local_gradient_bias_item
                    self.USER_bias_minibatch_accumulator[sample.user] += local_gradient_bias_user

                if self.use_embeddings:
                    for factor_index in range(self.n_factors):
                        if self.factors_dropout_mask[factor_index]:

                            # Copy original value to avoid messing up the updates
                            H_i = self.ITEM_factors[sample.item, factor_index]
                            W_u = self.USER_factors[sample.user, factor_index]

                            # Compute gradients
                            local_gradient_item = prediction_error * W_u - self.positive_reg * H_i
                            local_gradient_user = prediction_error * H_i - self.user_reg * W_u

                            # Store the gradient in the temporary accumulator
                            self.ITEM_factors_minibatch_accumulator[sample.item, factor_index] += local_gradient_item
                            self.USER_factors_minibatch_accumulator[sample.user, factor_index] += local_gradient_user

                    for factor_index in range(self.n_factors_user):
                        if self.factors_dropout_mask_user[factor_index]:

                            # Copy original value to avoid messing up the updates
                            H_i = self.ITEM_factors_user[sample.item, factor_index]
                            W_u = self.USER_factors_user[sample.user, factor_index]

                            #the coefficient of the second MF
                            coefficient_similarity_user=self.similarity_matrix_user[sample.user,:]
                            #coefficient_similarity_user=1

                            # Compute gradients
                            local_gradient_item_u = prediction_error * np.sum(coefficient_similarity_user) * W_u - self.item_reg_u * H_i
                            local_gradient_user_u = prediction_error * np.sum(coefficient_similarity_user) * H_i - self.user_reg_u * W_u

                            # Store the gradient in the temporary accumulator
                            self.ITEM_factors_minibatch_accumulator_user[sample.item, factor_index] += local_gradient_item_u
                            self.USER_factors_minibatch_accumulator_user[sample.user, factor_index] += local_gradient_user_u

                    for factor_index in range(self.n_factors_item):
                        if self.factors_dropout_mask_item[factor_index]:

                            # Copy original value to avoid messing up the updates
                            H_i = self.ITEM_factors_item[sample.item, factor_index]
                            W_u = self.USER_factors_item[sample.user, factor_index]

                            #the coefficient of the third MF
                            coefficient_similarity_item = self.similarity_matrix_item[sample.item,:]
                            #coefficient_similarity_user=1

                            # Compute gradients
                            local_gradient_item_i = prediction_error * np.sum(coefficient_similarity_item) * W_u - self.item_reg_i * H_i
                            local_gradient_user_i = prediction_error * np.sum(coefficient_similarity_item) * H_i - self.user_reg_i * W_u

                            # Store the gradient in the temporary accumulator
                            self.ITEM_factors_minibatch_accumulator_item[sample.item, factor_index] += local_gradient_item_i
                            self.USER_factors_minibatch_accumulator_item[sample.user, factor_index] += local_gradient_user_i



            self._apply_minibatch_updates_to_latent_factors()


            # Exponentiation of beta at the end of each mini batch
            if self.useAdam:

                self.beta_1_power_t *= self.beta_1
                self.beta_2_power_t *= self.beta_2

            processed_samples_last_print += 1

            if self.verbose and (processed_samples_last_print >= print_block_size or n_current_batch == n_total_batch-1):

                # Set block size to the number of items necessary in order to print every 300 seconds
                current_time = time.time()
                samples_per_sec = n_current_batch/(current_time - start_time_epoch)
                print_block_size = math.ceil(samples_per_sec * self.print_step_seconds)

                if current_time - last_print_time > self.print_step_seconds or n_current_batch == n_total_batch-1:
                    new_time_value, new_time_unit = seconds_to_biggest_unit(time.time() - start_time_epoch)

                    print("{}: Processed {} ({:4.1f}%) in {:.2f} {}. MSE loss {:.2E}. Sample per second: {:.0f}".format(
                        self.algorithm_name,
                        (n_current_batch+1)*self.batch_size,
                        100.0* (n_current_batch+1)/n_total_batch,
                        new_time_value, new_time_unit,
                        cumulative_loss/((n_current_batch+1)*self.batch_size),
                        float((n_current_batch+1)*self.batch_size) / (time.time() - start_time_epoch)))

                    last_print_time = current_time
                    processed_samples_last_print = 0

                    sys.stdout.flush()
                    sys.stderr.flush()




    def get_USER_factors(self):
        return np.array(self.USER_factors)


    def get_ITEM_factors(self):
        return np.array(self.ITEM_factors)

    def get_USER_factors_user(self):
        return np.array(self.USER_factors_user)


    def get_ITEM_factors_user(self):
        return np.array(self.ITEM_factors_user)

    def get_USER_factors_item(self):
        return np.array(self.USER_factors_item)


    def get_ITEM_factors_item(self):
        return np.array(self.ITEM_factors_item)

    def get_similarity_matrix_user(self):
        return np.array(self.similarity_matrix_user)


    def get_similarity_matrix_item(self):
        return np.array(self.similarity_matrix_item)


    def get_USER_bias(self):
        return np.array(self.USER_bias)


    def get_ITEM_bias(self):
        return np.array(self.ITEM_bias)


    def get_GLOBAL_bias(self):
        return np.array(self.GLOBAL_bias[0])



    def _init_minibatch_data_structures(self):

        # The shape depends on the batch size. 1 for FunkSVD 2 for BPR as it samples two items
        self.mini_batch_sampled_items = np.zeros(self.batch_size*2, dtype=np.int)
        self.mini_batch_sampled_users = np.zeros(self.batch_size, dtype=np.int)

        self.mini_batch_sampled_items_flag = np.zeros(self.n_items, dtype=np.int)
        self.mini_batch_sampled_users_flag = np.zeros(self.n_users, dtype=np.int)

        self.mini_batch_sampled_items_counter = 0
        self.mini_batch_sampled_users_counter = 0



    cdef void _clear_minibatch_data_structures(self):

        cdef long array_index, item_index

        for array_index in range(self.mini_batch_sampled_items_counter):
            item_index = self.mini_batch_sampled_items[array_index]
            self.mini_batch_sampled_items_flag[item_index] = False

        for array_index in range(self.mini_batch_sampled_users_counter):
            item_index = self.mini_batch_sampled_users[array_index]
            self.mini_batch_sampled_users_flag[item_index] = False

        self.mini_batch_sampled_items_counter = 0
        self.mini_batch_sampled_users_counter = 0



    cdef void _add_MSE_sample_in_minibatch(self, MSE_sample sample):

        if not self.mini_batch_sampled_items_flag[sample.item]:
            self.mini_batch_sampled_items_flag[sample.item] = True
            self.mini_batch_sampled_items[self.mini_batch_sampled_items_counter] = sample.item
            self.mini_batch_sampled_items_counter += 1

        if not self.mini_batch_sampled_users_flag[sample.user]:
            self.mini_batch_sampled_users_flag[sample.user] = True
            self.mini_batch_sampled_users[self.mini_batch_sampled_users_counter] = sample.user
            self.mini_batch_sampled_users_counter += 1


    cdef void _apply_minibatch_updates_to_latent_factors(self):

        cdef double local_gradient_item, local_gradient_user, local_gradient_bias_item, local_gradient_bias_user, local_gradient_bias_global
        cdef long sampled_user, sampled_item, n_sample_in_batch


        if self.use_bias:

            # Compute adaptive gradients
            local_gradient_bias_global = self.GLOBAL_bias_minibatch_accumulator[0] / self.batch_size
            local_gradient_bias_global = self.adaptive_gradient(local_gradient_bias_global, 0, 0, self.sgd_cache_bias_GLOBAL, self.sgd_cache_bias_GLOBAL_momentum_1, self.sgd_cache_bias_GLOBAL_momentum_2)

            # Apply updates to bias
            self.GLOBAL_bias[0] += self.learning_rate * local_gradient_bias_global
            self.GLOBAL_bias_minibatch_accumulator[0] = 0.0




        for n_sample_in_batch in range(self.mini_batch_sampled_items_counter):

            sampled_item = self.mini_batch_sampled_items[n_sample_in_batch]

            if self.use_bias:
                local_gradient_bias_item = self.ITEM_bias_minibatch_accumulator[sampled_item] / self.batch_size
                local_gradient_bias_item = self.adaptive_gradient(local_gradient_bias_item, sampled_item, 0, self.sgd_cache_bias_I, self.sgd_cache_bias_I_momentum_1, self.sgd_cache_bias_I_momentum_2)

                self.ITEM_bias[sampled_item] += self.learning_rate * local_gradient_bias_item
                self.ITEM_bias_minibatch_accumulator[sampled_item] = 0.0

            if self.use_embeddings:
                for factor_index in range(self.n_factors):
                    if self.factors_dropout_mask[factor_index]:
                        local_gradient_item = self.ITEM_factors_minibatch_accumulator[sampled_item, factor_index] / self.batch_size
                        local_gradient_item = self.adaptive_gradient(local_gradient_item, sampled_item, factor_index, self.sgd_cache_I, self.sgd_cache_I_momentum_1, self.sgd_cache_I_momentum_2)

                        self.ITEM_factors[sampled_item, factor_index] += self.learning_rate * local_gradient_item
                        self.ITEM_factors_minibatch_accumulator[sampled_item, factor_index] = 0.0

                for factor_index in range(self.n_factors_user):
                    if self.factors_dropout_mask_user[factor_index]:
                        local_gradient_item = self.ITEM_factors_minibatch_accumulator_user[sampled_item, factor_index] / self.batch_size
                        local_gradient_item = self.adaptive_gradient(local_gradient_item, sampled_item, factor_index, self.sgd_cache_I_user, self.sgd_cache_I_momentum_1_user, self.sgd_cache_I_momentum_2_user)

                        self.ITEM_factors_user[sampled_item, factor_index] += self.learning_rate_user * local_gradient_item
                        self.ITEM_factors_minibatch_accumulator_user[sampled_item, factor_index] = 0.0

                for factor_index in range(self.n_factors_item):
                    if self.factors_dropout_mask[factor_index]:
                        local_gradient_item = self.ITEM_factors_minibatch_accumulator_item[sampled_item, factor_index] / self.batch_size
                        local_gradient_item = self.adaptive_gradient(local_gradient_item, sampled_item, factor_index, self.sgd_cache_I_item, self.sgd_cache_I_momentum_1_item, self.sgd_cache_I_momentum_2_item)

                        self.ITEM_factors_item[sampled_item, factor_index] += self.learning_rate_item * local_gradient_item
                        self.ITEM_factors_minibatch_accumulator_item[sampled_item, factor_index] = 0.0





        for n_sample_in_batch in range(self.mini_batch_sampled_users_counter):

            sampled_user = self.mini_batch_sampled_users[n_sample_in_batch]

            if self.use_bias:
                local_gradient_bias_user = self.USER_bias_minibatch_accumulator[sampled_user] / self.batch_size
                local_gradient_bias_user = self.adaptive_gradient(local_gradient_bias_user, sampled_user, 0, self.sgd_cache_bias_U, self.sgd_cache_bias_U_momentum_1, self.sgd_cache_bias_U_momentum_2)

                self.USER_bias[sampled_user] += self.learning_rate * local_gradient_bias_user
                self.USER_bias_minibatch_accumulator[sampled_user] = 0.0

            if self.use_embeddings:
                for factor_index in range(self.n_factors):

                    if self.factors_dropout_mask[factor_index]:
                        local_gradient_user = self.USER_factors_minibatch_accumulator[sampled_user, factor_index] / self.batch_size
                        local_gradient_user = self.adaptive_gradient(local_gradient_user, sampled_user, factor_index, self.sgd_cache_U, self.sgd_cache_U_momentum_1, self.sgd_cache_U_momentum_2)

                        self.USER_factors[sampled_user, factor_index] += self.learning_rate * local_gradient_user
                        self.USER_factors_minibatch_accumulator[sampled_user, factor_index] = 0.0

                for factor_index in range(self.n_factors_user):

                    if self.factors_dropout_mask_user[factor_index]:
                        local_gradient_user = self.USER_factors_minibatch_accumulator_user[sampled_user, factor_index] / self.batch_size
                        local_gradient_user = self.adaptive_gradient(local_gradient_user, sampled_user, factor_index, self.sgd_cache_U_user, self.sgd_cache_U_momentum_1_user, self.sgd_cache_U_momentum_2_user)

                        self.USER_factors_user[sampled_user, factor_index] += self.learning_rate_user * local_gradient_user
                        self.USER_factors_minibatch_accumulator_user[sampled_user, factor_index] = 0.0

                for factor_index in range(self.n_factors_item):

                    if self.factors_dropout_mask_item[factor_index]:
                        local_gradient_user = self.USER_factors_minibatch_accumulator_item[sampled_user, factor_index] / self.batch_size
                        local_gradient_user = self.adaptive_gradient(local_gradient_user, sampled_user, factor_index, self.sgd_cache_U_item, self.sgd_cache_U_momentum_1_item, self.sgd_cache_U_momentum_2_item)

                        self.USER_factors_item[sampled_user, factor_index] += self.learning_rate_item * local_gradient_user
                        self.USER_factors_minibatch_accumulator_item[sampled_user, factor_index] = 0.0





    cdef double adaptive_gradient(self, double gradient, long user_or_item_id, long factor_id, double[:,:] sgd_cache, double[:,:] sgd_cache_momentum_1, double[:,:] sgd_cache_momentum_2):


        cdef double gradient_update

        if self.useAdaGrad:
            sgd_cache[user_or_item_id, factor_id] += gradient ** 2

            gradient_update = gradient / (sqrt(sgd_cache[user_or_item_id, factor_id]) + 1e-8)


        elif self.useRmsprop:
            sgd_cache[user_or_item_id, factor_id] = sgd_cache[user_or_item_id, factor_id] * self.gamma + (1 - self.gamma) * gradient ** 2

            gradient_update = gradient / (sqrt(sgd_cache[user_or_item_id, factor_id]) + 1e-8)


        elif self.useAdam:

            sgd_cache_momentum_1[user_or_item_id, factor_id] = \
                sgd_cache_momentum_1[user_or_item_id, factor_id] * self.beta_1 + (1 - self.beta_1) * gradient

            sgd_cache_momentum_2[user_or_item_id, factor_id] = \
                sgd_cache_momentum_2[user_or_item_id, factor_id] * self.beta_2 + (1 - self.beta_2) * gradient**2


            self.momentum_1 = sgd_cache_momentum_1[user_or_item_id, factor_id]/ (1 - self.beta_1_power_t)
            self.momentum_2 = sgd_cache_momentum_2[user_or_item_id, factor_id]/ (1 - self.beta_2_power_t)

            gradient_update = self.momentum_1/ (sqrt(self.momentum_2) + 1e-8)


        else:

            gradient_update = gradient



        return gradient_update




    cdef MSE_sample sampleMSE_Cython(self):

        cdef MSE_sample sample = MSE_sample(-1,-1,-1.0)
        cdef long index, start_pos_seen_items, end_pos_seen_items

        cdef int neg_item_selected, sample_positive, n_seen_items = 0

        # Skip users with no interactions or with no negative items
        while n_seen_items == 0 or n_seen_items == self.n_items:

            sample.user = rand() % self.n_users

            start_pos_seen_items = self.URM_train_indptr[sample.user]
            end_pos_seen_items = self.URM_train_indptr[sample.user+1]

            n_seen_items = end_pos_seen_items - start_pos_seen_items


        # Decide to sample positive or negative
        if self.MSE_sample_negative_interactions_flag:
            sample_positive = rand() <= self.MSE_negative_interactions_quota * RAND_MAX
        else:
            sample_positive = True


        if sample_positive:

            # Sample positive
            index = rand() % n_seen_items

            sample.item = self.URM_train_indices[start_pos_seen_items + index]
            sample.rating = self.URM_train_data[start_pos_seen_items + index]

        else:

            # Sample negative
            neg_item_selected = False

            # It's faster to just try again then to build a mapping of the non-seen items
            # for every user
            while not neg_item_selected:

                sample.item = rand() % self.n_items
                sample.rating = 0.0

                index = 0
                # Indices data is sorted, so I don't need to go to the end of the current row
                while index < n_seen_items and self.URM_train_indices[start_pos_seen_items + index] < sample.item:
                    index+=1

                # If the positive item in position 'index' is == sample.item, negative not selected
                # If the positive item in position 'index' is > sample.item or index == n_seen_items, negative selected
                if index == n_seen_items or self.URM_train_indices[start_pos_seen_items + index] > sample.item:
                    neg_item_selected = True



        return sample