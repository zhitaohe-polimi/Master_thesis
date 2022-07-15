"""
Created on 17/09/17

@author: Maurizio Ferrari Dacrema
"""

#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
import time
import sys

from libc.math cimport exp, sqrt
from libc.stdlib cimport rand, RAND_MAX

from cpython.array cimport array, clone


cdef struct Prediction:
    double weighted
    double weighted_gradient


cdef struct SGD_minibatch_update_s:

    long[:] gradient_weights_update_mask
    long[:] gradient_weights_update_index
    long gradient_weights_update_counter

    long[:] gradient_users_update_mask
    long[:] gradient_users_update_index
    long gradient_users_update_counter




cdef class FW_RATING_RMSE_Cython_Epoch:

    cdef double[:,:] S_weighted_gradient, S_denominator, S_weighted
    cdef double[:] bias_user, bias_item
    cdef double[:] weights, gradientWeights
    cdef int n_features, n_items
    cdef int topK
    cdef int useBias
    cdef int useAdaGrad
    cdef int useLinearWeights
    cdef int batch_size
    cdef int shrink
    cdef long numPositiveIteractions

    cdef URM_train, ICM, FTI_M

    cdef float learning_rate, lambda_l1, lambda_l2
    cdef int[:] ICM_indices, ICM_indptr
    cdef int[:] FTI_M_indices, FTI_M_indptr
    cdef int[:] URM_train_indices, URM_train_indptr







    def __init__(self, ICM, URM_train, weights, S_denominator, S_weighted_gradient, S_weighted,
                 learning_rate = 0.05, lambda_l1 = 0.0, lambda_l2 = 0.0, shrink = 100,
                 topK=False, useLinearWeights=True, useBias=False, batch_size=1, sgd_mode='adagrad'):

        super(FW_RATING_RMSE_Cython_Epoch, self).__init__()

        self.URM_train = URM_train
        self.ICM = ICM
        self.useBias = useBias

        self.S_denominator = S_denominator
        self.S_weighted_gradient = S_weighted_gradient
        self.S_weighted = S_weighted

        self.ICM_indices = ICM.indices
        self.ICM_indptr = ICM.indptr
        self.FTI_M_indices = ICM.transpose().tocsr().indices
        self.FTI_M_indptr = ICM.transpose().tocsr().indptr




        self.numPositiveIteractions = int(URM_train.nnz*0.3)
        self.URM_train_indices = URM_train.indices
        self.URM_train_indptr = URM_train.indptr

        self.weights = weights

        if self.useBias:
            self.bias_user = np.zeros((self.URM_train.shape[0]))
            self.bias_item = np.zeros((self.URM_train.shape[1]))



        self.shrink = shrink
        self.n_features = len(weights)
        self.n_items = ICM.shape[0]
        self.topK = topK
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.useLinearWeights = useLinearWeights



        if sgd_mode=='adagrad':
            self.useAdaGrad = True
        elif sgd_mode=='sgd':
            pass
        else:
            raise ValueError(
                "SGD_mode not valid. Acceptable values are: 'sgd', 'adagrad'. Provided value was '{}'".format(
                    sgd_mode))






    cdef double[:] computeItemSimilarities(self, long item_id):

        # Create template to create an array initialized with zeros
        cdef array[double] template_zero = array('d')
        cdef array[double] result = clone(template_zero, self.n_items, zero=True)

        #cdef double[:] result = np.zeros((self.n_items))

        cdef int[:] items_containing_feature
        cdef int[:] item_features = self.getItemFeatures(item_id)

        cdef long feature_index, feature_id, item_index_to_update, item_id_to_update


        for feature_index in range(len(item_features)):

            feature_id = item_features[feature_index]

            if self.weights[feature_id] > 0.0:

                items_containing_feature = self.FTI_M_indices[self.FTI_M_indptr[feature_id]:self.FTI_M_indptr[feature_id+1]]

                for item_index_to_update in range(len(items_containing_feature)):

                    item_id_to_update = items_containing_feature[item_index_to_update]

                    result[item_id_to_update] += self.weights[feature_id]

        return result
    #
    #
    # def epochIterationNoBatch_Cython(self):
    #
    #     cdef Prediction prediction
    #     cdef double delta_b, delta_w
    #     cdef int[:] item_features
    #     cdef double[:] S_weighted, S_weighted_gradient
    #     cdef double[:] sgd_cache_weights, sgd_cache_bias_user
    #
    #     # Get number of available interactions
    #     cdef long numCurrentSample, index, weight_index
    #     cdef long numSamples = int(self.numPositiveIteractions)
    #
    #     # Shuffle data
    #     cdef long[:] newOrdering = np.arange(numSamples)
    #     np.random.shuffle(newOrdering)
    #
    #     URM_train_coo = self.URM_train.tocoo()
    #
    #     cdef int[:] user_id_newOrdering = URM_train_coo.row[newOrdering]
    #     cdef int[:] item_id_newOrdering = URM_train_coo.col[newOrdering]
    #     cdef float[:] rating_newOrdering = URM_train_coo.data[newOrdering]
    #
    #     cdef long user_id, item_id
    #     cdef float rating
    #
    #     # Initialize cache
    #     if self.useAdaGrad:
    #         sgd_cache_weights = np.zeros_like(self.weights)
    #         sgd_cache_bias_user = np.zeros_like(self.bias_user)
    #
    #
    #     start_time_epoch = time.time()
    #     start_time_batch = time.time()
    #
    #     for numCurrentSample in range(numSamples):
    #
    #         user_id = user_id_newOrdering[numCurrentSample]
    #         item_id = item_id_newOrdering[numCurrentSample]
    #         rating = rating_newOrdering[numCurrentSample]
    #
    #         # Compute only required item similarity
    #         S_weighted = self.computeItemSimilarities(item_id)
    #         S_weighted_gradient = self.S_weighted_gradient[item_id, :]
    #
    #         for index in range(self.n_items):
    #             S_weighted[index] = S_weighted[index] * self.S_denominator[item_id, index]
    #
    #         prediction = self.computePrediction(user_id, S_weighted, S_weighted_gradient)
    #         prediction.weighted -= rating
    #
    #         delta_b = prediction.weighted
    #         delta_w =  delta_b * prediction.weighted_gradient
    #
    #         item_features = self.getItemFeatures(item_id)
    #
    #         for index in range(len(item_features)):
    #
    #             weight_index = item_features[index]
    #
    #             # Apply L1 and L2 regularization
    #             localGradient = delta_w + self.lambda_l1 + 2 * self.weights[weight_index] * self.lambda_l2
    #
    #             if self.useAdaGrad:
    #                 sgd_cache_weights[weight_index] += localGradient ** 2
    #
    #                 self.weights[weight_index] -= self.learning_rate * localGradient / (sqrt(sgd_cache_weights[weight_index]) + 1e-8)
    #
    #             else:
    #                 self.weights[weight_index] -= self.learning_rate * delta_w
    #
    #
    #             # Clamp weight value to ensure it is between 0.0 and 1.0
    #             if self.weights[weight_index]>1.0:
    #                 self.weights[weight_index] = 1.0
    #             elif self.weights[weight_index]<0.0:
    #                 self.weights[weight_index] = 0.0
    #
    #         if self.useBias:
    #
    #             if self.useAdaGrad:
    #                 sgd_cache_bias_user[user_id] += delta_b ** 2
    #
    #                 self.bias_user[user_id] -= self.learning_rate * delta_b / (sqrt(sgd_cache_bias_user[user_id]) + 1e-8)
    #
    #             else:
    #                 self.bias_user[user_id] -= self.learning_rate * delta_b
    #
    #
    #
    #             # Clamp bias value to ensure it is between +5.0 and -5.0
    #             if self.bias_user[user_id] < -5.0:
    #                 self.bias_user[user_id] = -5.0
    #             elif self.bias_user[user_id] > 5.0:
    #                 self.bias_user[user_id] = 5.0
    #
    #
    #         if (numCurrentSample % 100000 == 0 and numCurrentSample != 0) or numCurrentSample == numSamples - 1:
    #             print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
    #                 numCurrentSample,
    #                 100.0 * float(numCurrentSample) / numSamples,
    #                 time.time() - start_time_batch,
    #                 float(numCurrentSample + 1) / (time.time() - start_time_epoch)))
    #
    #             # Flush buffer to ensure update progress is written on linux nohup file
    #             sys.stdout.flush()
    #             sys.stderr.flush()
    #
    #             start_time_batch = time.time()
    #
    #
    #     return np.array(self.weights)


    def epochIterationBoth_Cython(self):

        cdef Prediction prediction
        cdef double delta_b, delta_w
        cdef int[:] item_features
        cdef double[:] S_weighted, S_weighted_gradient
        cdef double[:] sgd_cache_weights, sgd_cache_bias_user, sgd_cache_bias_item


        cdef long[:] gradient_weights_update_mask = np.zeros_like(self.weights, dtype=np.int)
        cdef long[:] gradient_weights_update_index = np.zeros_like(self.weights, dtype=np.int)
        cdef long gradient_weights_update_counter

        cdef long[:] gradient_users_update_mask = np.zeros_like(self.bias_user, dtype=np.int)
        cdef long[:] gradient_users_update_index = np.zeros_like(self.bias_user, dtype=np.int)
        cdef long gradient_users_update_counter

        cdef long[:] gradient_items_update_mask = np.zeros_like(self.bias_item, dtype=np.int)
        cdef long[:] gradient_items_update_index = np.zeros_like(self.bias_item, dtype=np.int)
        cdef long gradient_items_update_counter


        # Get number of available interactions
        cdef long numCurrentSample, index, weight_index, user_index, item_index
        cdef long numSamples = int(self.numPositiveIteractions)

        # Shuffle data
        cdef long[:] newOrdering = np.arange(numSamples)
        np.random.shuffle(newOrdering)

        URM_train_coo = self.URM_train.tocoo()

        cdef int[:] user_id_newOrdering = URM_train_coo.row[newOrdering]
        cdef int[:] item_id_newOrdering = URM_train_coo.col[newOrdering]
        cdef float[:] rating_newOrdering = URM_train_coo.data[newOrdering]

        cdef long user_id, item_id
        cdef float rating

        # Initialize cache
        if self.useAdaGrad:
            sgd_cache_weights = np.zeros_like(self.weights)
            sgd_cache_bias_user = np.zeros_like(self.bias_user)
            sgd_cache_bias_item = np.zeros_like(self.bias_item)



        start_time_epoch = time.time()
        start_time_batch = time.time()

        delta_b = 0.0
        delta_w = 0.0
        gradient_weights_update_counter = 0
        gradient_users_update_counter = 0
        gradient_items_update_counter = 0


        for numCurrentSample in range(numSamples):

            user_id = user_id_newOrdering[numCurrentSample]
            item_id = item_id_newOrdering[numCurrentSample]
            rating = rating_newOrdering[numCurrentSample]

            # Compute only required item similarity
            S_weighted = self.computeItemSimilarities(item_id)
            S_weighted_gradient = self.S_weighted_gradient[item_id, :]

            for index in range(self.n_items):
                S_weighted[index] = S_weighted[index] * self.S_denominator[item_id, index]

            prediction = self.computePrediction(user_id, item_id, S_weighted, S_weighted_gradient)
            prediction.weighted -= rating

            delta_b += prediction.weighted
            delta_w +=  delta_b * prediction.weighted_gradient

            item_features = self.getItemFeatures(item_id)



            for index in range(len(item_features)):

                weight_index = item_features[index]

                # Keep track of which weights will have to be updated
                if gradient_weights_update_mask[weight_index] == 0:
                    # update_mask contains flags, if True the weight is already queued for update
                    gradient_weights_update_mask[weight_index] = 1
                    # update_index contains the weight index, it acts as a queue, new weights are added at the end
                    gradient_weights_update_index[gradient_weights_update_counter] = weight_index
                    # update_counter keeps track of how many weighs are in update_index
                    gradient_weights_update_counter += 1


            if gradient_users_update_mask[user_id] == 0:
                # update_mask contains flags, if True the user is already queued for update
                gradient_users_update_mask[user_id] = 1
                # update_index contains the user index, it acts as a queue, new user are added at the end
                gradient_users_update_index[gradient_users_update_counter] = user_id
                # update_counter keeps track of how many user are in update_index
                gradient_users_update_counter += 1


            if gradient_items_update_mask[item_id] == 0:
                # update_mask contains flags, if True the user is already queued for update
                gradient_items_update_mask[item_id] = 1
                # update_index contains the user index, it acts as a queue, new user are added at the end
                gradient_items_update_index[gradient_items_update_counter] = item_id
                # update_counter keeps track of how many user are in update_index
                gradient_items_update_counter += 1



            # If it is time to update: always if no batch, every batch_size samples otherwise; update
            if numCurrentSample % self.batch_size == 0:

                # If no batch, the gradient remains identical, otherwise is averaged
                delta_b = delta_b / self.batch_size
                delta_w = delta_w / self.batch_size


                # Now update the required weights
                for index in range(gradient_weights_update_counter):

                    weight_index = gradient_weights_update_index[index]

                    # Clean for next iteration
                    # update_mask and update_index requires different indices
                    gradient_weights_update_mask[weight_index] = 0
                    gradient_weights_update_index[index] = 0

                    # Apply L1 and L2 regularization
                    localGradient = delta_w + self.lambda_l1 + 2 * self.weights[weight_index] * self.lambda_l2

                    if self.useAdaGrad:
                        sgd_cache_weights[weight_index] += localGradient ** 2

                        self.weights[weight_index] -= self.learning_rate * localGradient / (sqrt(sgd_cache_weights[weight_index]) + 1e-8)

                    else:
                        self.weights[weight_index] -= self.learning_rate * delta_w


                    # Clamp weight value to ensure it is between 0.0 and 1.0
                    if self.weights[weight_index]>1.0:
                        self.weights[weight_index] = 1.0
                    elif self.weights[weight_index]<0.0:
                        self.weights[weight_index] = 0.0


                if self.useBias:

                    # Now update the required weights
                    for index in range(gradient_users_update_counter):

                        user_index = gradient_users_update_index[index]

                        # Clean for next iteration
                        # update_mask and update_index requires different indices
                        gradient_users_update_mask[user_index] = 0
                        gradient_users_update_index[index] = 0

                        if self.useAdaGrad:
                            sgd_cache_bias_user[user_index] += delta_b ** 2

                            self.bias_user[user_index] -= self.learning_rate * delta_b / (sqrt(sgd_cache_bias_user[user_index]) + 1e-8)

                        else:
                            self.bias_user[user_index] -= self.learning_rate * delta_b

                    # Clamp bias value to ensure it is between +5.0 and -5.0
                    if self.bias_user[user_index] < -5.0:
                        self.bias_user[user_index] = -5.0
                    elif self.bias_user[user_index] > 5.0:
                        self.bias_user[user_index] = 5.0


                    # Now update the required weights
                    for index in range(gradient_items_update_counter):

                        item_index = gradient_items_update_index[index]

                        # Clean for next iteration
                        # update_mask and update_index requires different indices
                        gradient_items_update_mask[item_index] = 0
                        gradient_items_update_index[index] = 0

                        if self.useAdaGrad:
                            sgd_cache_bias_item[item_index] += delta_b ** 2

                            self.bias_item[item_index] -= self.learning_rate * delta_b / (sqrt(sgd_cache_bias_item[item_index]) + 1e-8)

                        else:
                            self.bias_item[item_index] -= self.learning_rate * delta_b

                    # Clamp bias value to ensure it is between +5.0 and -5.0
                    if self.bias_item[item_index] < -5.0:
                        self.bias_item[item_index] = -5.0
                    elif self.bias_item[item_index] > 5.0:
                        self.bias_item[item_index] = 5.0


                delta_b = 0.0
                delta_w = 0.0
                gradient_weights_update_counter = 0
                gradient_users_update_counter = 0
                gradient_items_update_counter = 0


            if (numCurrentSample % 100000 == 0 and numCurrentSample != 0) or numCurrentSample == numSamples - 1:
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    numCurrentSample,
                    100.0 * float(numCurrentSample) / numSamples,
                    time.time() - start_time_batch,
                    float(numCurrentSample + 1) / (time.time() - start_time_epoch)))

                # Flush buffer to ensure update progress is written on linux nohup file
                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()


        return np.array(self.weights)







    # Using memoryview instead of the sparse matrix itself allows for much faster access
    cdef int[:] getCommonFeatures(self, long index):
        return self.commonFeatures_indices[self.commonFeatures_indptr[index]:self.commonFeatures_indptr[index + 1]]

    cdef int[:] getItemFeatures(self, long item_id):
        return self.ICM_indices[self.ICM_indptr[item_id]:self.ICM_indptr[item_id+1]]


    cdef int[:] getUserSeenItems(self, long user_id):
        return self.URM_train_indices[self.URM_train_indptr[user_id]:self.URM_train_indptr[user_id+1]]


    cdef Prediction computePrediction(self, long user_id, long item_id, double[:] weights_vector, double[:] weights_gradient_vector):

        cdef Prediction prediction = Prediction(0.0, 0.0)
        cdef long index, item_index
        cdef int[:] user_seen_items = self.getUserSeenItems(user_id)

        for index in range(len(user_seen_items)):
            item_index = user_seen_items[index]

            prediction.weighted += weights_vector[item_index]
            prediction.weighted_gradient += weights_gradient_vector[item_index]

        if self.useBias:
            prediction.weighted += self.bias_user[user_id]
            prediction.weighted_gradient += self.bias_user[user_id]

            prediction.weighted += self.bias_item[item_id]
            prediction.weighted_gradient += self.bias_item[item_id]


        return prediction
