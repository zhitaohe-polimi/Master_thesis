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



from libc.math cimport exp, sqrt
from libc.stdlib cimport rand, RAND_MAX


cdef struct BPR_sample:
    long user
    long pos_item
    long neg_item


cdef struct Prediction:
    double weighted
    double weighted_gradient




cdef class FW_RATING_BPR_Cython_Epoch:

    cdef double[:,:] S_weighted_gradient, S_denominator, S_weighted
    cdef double[:] bias_user, bias_item
    cdef double[:] weights, gradientWeights
    cdef int n_features, n_items, n_users
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


    cdef long[:] eligibleUsers
    cdef long numEligibleUsers


    cdef int[:] seenItemsSampledUser
    cdef int numSeenItemsSampledUser

    def __init__(self, ICM, URM_train, weights, S_denominator, S_weighted_gradient, S_weighted, eligibleUsers,
                 learning_rate = 0.05, lambda_l1 = 0.0, lambda_l2 = 0.0, shrink = 100,
                 topK=False, useLinearWeights=True, useBias=False, batch_size=1, sgd_mode='adagrad'):

        super(FW_RATING_BPR_Cython_Epoch, self).__init__()

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



        self.eligibleUsers = eligibleUsers
        self.numEligibleUsers = len(eligibleUsers)


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
        self.n_users = self.URM_train.shape[0]
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






    def epochIterationBoth_Cython(self):

        cdef Prediction prediction_i, prediction_j, prediction_ij
        cdef double gradient, localGradient
        cdef int[:] item_features
        cdef double[:] S_weighted, S_weighted_gradient
        cdef double[:] sgd_cache_weights


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

        cdef long user_id, item_id, index, feature_index, feature_id
        cdef float rating

        # Initialize cache
        if self.useAdaGrad:
            sgd_cache_weights = np.zeros_like(self.weights)



        start_time_epoch = time.time()
        start_time_batch = time.time()

        prediction_ij = Prediction(0.0, 0.0)


        for numCurrentSample in range(numSamples):

            sample = self.sampleBatch_Cython()

            user_id = sample.user
            item_id_i = sample.pos_item
            item_id_j = sample.neg_item


            # Compute only required item similarity
            S_weighted = self.computeItemSimilarities(item_id_i)
            S_weighted_gradient = self.S_weighted_gradient[item_id_i, :]

            for index in range(self.n_items):
                S_weighted[index] = S_weighted[index] * self.S_denominator[item_id_i, index]

            prediction_i = self.computePrediction(user_id, item_id_i, S_weighted, S_weighted_gradient)

            # Compute only required item similarity
            S_weighted = self.computeItemSimilarities(item_id_j)
            S_weighted_gradient = self.S_weighted_gradient[item_id_j, :]

            for index in range(self.n_items):
                S_weighted[index] = S_weighted[index] * self.S_denominator[item_id_j, index]

            prediction_j = self.computePrediction(user_id, item_id_j, S_weighted, S_weighted_gradient)



            prediction_ij.weighted = prediction_i.weighted - prediction_j.weighted
            prediction_ij.weighted_gradient = prediction_i.weighted_gradient - prediction_j.weighted_gradient


            gradient = prediction_ij.weighted * prediction_ij.weighted_gradient
            gradient = 1 / (1 + exp(gradient))


            index = 0
            while index < self.numSeenItemsSampledUser:

                seenItem = self.seenItemsSampledUser[index]
                index +=1

                item_features = self.getItemFeatures(seenItem)

                feature_index = 0
                while feature_index < len(item_features):

                    weight_index = item_features[feature_index]
                    feature_index += 1

                    localGradient = gradient

                    if self.useAdaGrad:
                        sgd_cache_weights[weight_index] += gradient ** 2

                        localGradient =  gradient / (sqrt(sgd_cache_weights[weight_index]) + 1e-8)


                    if seenItem != item_id_i:
                        self.weights[weight_index] += self.learning_rate * localGradient

                    if seenItem != item_id_j:
                        self.weights[weight_index] -= self.learning_rate * localGradient


                    # Clamp weight value to ensure it is between 0.0 and 1.0
                    if self.weights[weight_index]>1.0:
                        self.weights[weight_index] = 1.0
                    elif self.weights[weight_index]<0.0:
                        self.weights[weight_index] = 0.0



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



    cdef BPR_sample sampleBatch_Cython(self):

        cdef BPR_sample sample = BPR_sample()
        cdef long index
        cdef int negItemSelected



        # Warning: rand() returns an integer, in order to avoid integer division we force the
        # denominator to be a float
        cdef double RAND_MAX_DOUBLE = RAND_MAX


        index = int(rand() / RAND_MAX_DOUBLE * self.numEligibleUsers )


        sample.user = self.eligibleUsers[index]

        self.seenItemsSampledUser = self.getUserSeenItems(sample.user)
        self.numSeenItemsSampledUser = len(self.seenItemsSampledUser)

        index = int(rand() / RAND_MAX_DOUBLE * self.numSeenItemsSampledUser )


        sample.pos_item = self.seenItemsSampledUser[index]


        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        # for every user
        while (not negItemSelected):
            sample.neg_item = int(rand() / RAND_MAX_DOUBLE  * self.n_items )

            index = 0
            while index < self.numSeenItemsSampledUser and self.seenItemsSampledUser[index]!=sample.neg_item:
                index+=1

            if index == self.numSeenItemsSampledUser:
                negItemSelected = True

        #print(sample)

        return sample






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


        return prediction
