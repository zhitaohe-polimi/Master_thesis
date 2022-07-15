"""
Created on 31/03/18

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

# PyMem malloc and free are slightly faster than plain C equivalents as they optimize OS calls
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from libc.math cimport exp, sqrt
from libc.stdlib cimport rand, RAND_MAX



from Recommenders.Recommender_utils import check_matrix

import time, sys
import scipy.sparse as sps





cdef struct BPR_sample:
    long user
    long pos_item
    long neg_item
    long seen_items_start_pos
    long seen_items_end_pos



cdef class SLIM_Structure_Cython_Epoch:


    cdef int[:] S_indices, S_indptr
    cdef int[:] ICM_indices, ICM_indptr
    cdef double[:] S_data

    cdef double[:,:] S_dense

    cdef int[:] URM_indices, URM_indptr
    cdef double[:] URM_data
    cdef double[:] user_bias

    cdef int[:] URM_indices_csc, URM_indptr_csc
    cdef double[:] URM_data_csc

    cdef double learning_rate, lambda_1, lambda_2

    cdef int n_items, n_users, n_features

    cdef int topK, batch_size

    # Structure
    cdef int structure_full, structure_common_feature, structure_similarity
    cdef int[:] common_feature_mask, common_feature_item_id


    # Adaptive gradient
    cdef int useAdaGrad, useRmsprop, useAdam

    cdef double [:] sgd_cache
    cdef double gamma

    cdef double [:] sgd_cache_momentum_1, sgd_cache_momentum_2
    cdef double beta_1, beta_2, beta_1_power_t, beta_2_power_t
    cdef double momentum_1, momentum_2


    def __init__(self, URM, S_structure = None, ICM = None, topK = 100, learning_rate=0.001, batch_size = 1,
                 lambda_1=0.001, lambda_2=0.001,
                 init_type = "random", structure_mode = "full",
                 sgd_mode='adam', gamma=0.995, beta_1=0.9, beta_2=0.999):
        """

        :param S_structure:
        :param URM:
        :param learning_rate:
        :param b:
        :param g:
        :param init_type:
        """

        #python compileCython.py SLIM_Structure_Cython_Epoch.pyx build_ext --inplace
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2


        URM = check_matrix(URM, "csr")
        self.URM_indices = np.array(URM.indices, dtype=np.int32)
        self.URM_indptr = np.array(URM.indptr, dtype=np.int32)
        self.URM_data = np.array(URM.data, dtype=np.float64)

        self.user_bias = np.array(URM.mean(axis=1), dtype=np.float64).ravel()

        URM = check_matrix(URM, "csc")
        self.URM_indices_csc = np.array(URM.indices, dtype=np.int32)
        self.URM_indptr_csc = np.array(URM.indptr, dtype=np.int32)
        self.URM_data_csc = np.array(URM.data, dtype=np.float64)


        self.n_users = URM.shape[0]
        self.n_items = URM.shape[1]


        self.structure_full = False
        self.structure_similarity = False
        self.structure_common_feature = False

        if structure_mode == "full":
            self.structure_full = True

            self.topK = topK

            if self.topK > self.n_items:
                self.topK = self.n_items

            if self.topK == False:

                if init_type == "copy_similarity":
                    self.S_dense = np.array(S_structure.toarray(), dtype=np.float64)
                elif init_type == "random":
                    self.S_dense = np.random.normal(0.001, 0.1, (self.n_items,self.n_items)).astype(np.float64)
                elif init_type == "one":
                    self.S_dense = np.ones((self.n_items,self.n_items), dtype=np.float64)
                elif init_type == "zero":
                    self.S_dense = np.zeros((self.n_items,self.n_items), dtype=np.float64)
                else:
                    raise ValueError("SLIM_Structure_Cython_Epoch: 'init_type' not recognized")

            else:

                self.S_indices = np.zeros(self.n_items*self.topK, dtype=np.int32)

                # Indptr contais the position at which an item begins, always topK steps
                self.S_indptr = np.zeros(self.n_items + 1, dtype=np.int32)

                for item_id in range(self.n_items):
                    self.S_indptr[item_id] = item_id*self.topK

                self.S_indptr[self.n_items] = len(self.S_indices)


                if init_type == "copy_similarity":
                    self.S_data = np.array(self.n_items*self.topK, dtype=np.float64)
                elif init_type == "random":
                    self.S_data = np.random.normal(0.001, 0.1, (self.n_items*self.topK)).astype(np.float64)
                elif init_type == "one":
                    self.S_data = np.ones(self.n_items*self.topK, dtype=np.float64)
                elif init_type == "zero":
                    self.S_data = np.zeros(self.n_items*self.topK, dtype=np.float64)
                else:
                    raise ValueError("SLIM_Structure_Cython_Epoch: 'init_type' not recognized")
        #
        # elif structure_mode == "common_feature":
        #
        #     if ICM is None:
        #         raise ValueError("SLIM_Structure_Cython_Epoch: 'structure_mode' is 'common_feature' but no 'ICM' was provided")
        #
        #     self.structure_common_feature = True
        #
        #     ICM = check_matrix(ICM, "csr")
        #     self.ICM_indices = np.array(ICM.indices, dtype=np.int32)
        #     self.ICM_indptr = np.array(ICM.indptr, dtype=np.int32)
        #
        #     self.n_features = ICM.shape[1]
        #     self.common_feature_mask = np.zeros(self.n_features, dtype=np.int32)
        #

        elif structure_mode == "similarity":

            if S_structure is None:
                raise ValueError("SLIM_Structure_Cython_Epoch: 'structure_mode' is 'similarity' but no 'S_structure' was provided")

            self.structure_similarity = True

            S_structure = check_matrix(S_structure, "csr")
            self.S_indices = np.array(S_structure.indices, dtype=np.int32)
            self.S_indptr = np.array(S_structure.indptr, dtype=np.int32)

            if init_type == "copy_similarity":
                self.S_data = np.array(S_structure.data, dtype=np.float64)
            elif init_type == "random":
                self.S_data = np.random.normal(0.001, 0.1, (len(S_structure.data))).astype(np.float64)
            elif init_type == "one":
                self.S_data = np.ones_like(S_structure.data, dtype=np.float64)
            elif init_type == "zero":
                self.S_data = np.zeros_like(S_structure.data, dtype=np.float64)
            else:
                raise ValueError("SLIM_Structure_Cython_Epoch: 'init_type' not recognized")

        else:

            raise ValueError("SLIM_Structure_Cython_Epoch: 'structure_mode' not recognized")





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

        elif sgd_mode=='sgd':
            pass
        else:
            raise ValueError(
                "SGD_mode not valid. Acceptable values are: 'sgd', 'adagrad', 'rmsprop', 'adam'. Provided value was '{}'".format(
                    sgd_mode))




    def epochIteration_Cython(self, epochs=30, loss = "mse", force_positive = False, sample_quota = 0.10):

        if loss == "mse":
            print("LOSS MSE")
            self.epochIteration_Cython_SGD(epochs, False, force_positive, sample_quota)

        elif loss == "bpr":
            print("LOSS BPR")
            self.epochIteration_Cython_SGD(epochs, True, force_positive, sample_quota)
        else:
            raise ValueError("Loss value not recognized")



    cdef init_S_structure_current_item(self, int current_item, int* Si_mask, double* Si_dense):
        """
        The function initializes the dense similarity vector for current item from the global data structure
        IMPORTANT: the similarity matrix has similarities for item i in row i
        """

        cdef int item_index, item_id
        cdef int feature_index, feature_id

        cdef int[:] S_current_item_indices
        cdef double[:] S_current_item_data

        cdef int[:] current_item_features, other_item_features
        cdef int has_a_common_feature

        #
        # if self.structure_common_feature:
        #
        #     # Accept all items having at least a common feature
        #     S_current_item_data = np.random.normal(0.001, 0.1, (self.n_items)).astype(np.float64)
        #
        #     # Set current item features
        #
        #     current_item_features = self.ICM_indices[self.ICM_indptr[current_item]:self.ICM_indptr[current_item + 1]]
        #
        #     for feature_index in range(len(current_item_features)):
        #         feature_id = current_item_features[feature_index]
        #
        #         self.common_feature_mask[feature_id] = True
        #
        #
        #     for item_id in range(self.n_items):
        #
        #         other_item_features = self.ICM_indices[self.ICM_indptr[item_id]:self.ICM_indptr[item_id + 1]]
        #
        #         has_a_common_feature = False
        #
        #         # Check its features against the ones of the current_item
        #         for feature_index in range(len(other_item_features)):
        #             feature_id = current_item_features[feature_index]
        #
        #             # As soon as I find one, terminate
        #             if self.common_feature_mask[feature_id]:
        #                 has_a_common_feature = True
        #                 break
        #
        #
        #         if has_a_common_feature:
        #             Si_mask[item_id] = True
        #             Si_dense[item_id] = S_current_item_data[item_id]
        #
        #
        #     # Clean feature_mask
        #     for feature_index in range(len(current_item_features)):
        #         feature_id = current_item_features[feature_index]
        #
        #         self.common_feature_mask[feature_id] = False



        if self.structure_full and not self.topK:


            for item_id in range(self.n_items):

                Si_mask[item_id] = True
                Si_dense[item_id] = self.S_dense[current_item, item_id]


        else:
            # Structure_similarity and structure_dense topK behave in a sligthly different way

            S_current_item_indices = self.S_indices[self.S_indptr[current_item]:self.S_indptr[current_item + 1]]
            S_current_item_data = self.S_data[self.S_indptr[current_item]:self.S_indptr[current_item + 1]]


            # If structure full, all mask cells must be true, not only those associated to a
            # neighbor value
            if self.structure_full and self.topK:

                for item_id in range(self.n_items):

                    Si_mask[item_id] = True
                    Si_dense[item_id] = 0.0


            # If structure similarity, only this specific set of cells is available
            for item_index in range(len(S_current_item_indices)):

                item_id = S_current_item_indices[item_index]

                Si_mask[item_id] = True
                Si_dense[item_id] = S_current_item_data[item_index]





    cdef clear_and_save_S_structure_current_item(self, int current_item, int* Si_mask, double* Si_dense):
        """
        The function copies the similarities of the current item into the final data structure
        IMPORTANT: the similarity matrix has similarities for item i in row i
        """

        cdef int item_index, item_id
        cdef int num_S_items, start_pos

        cdef int[:] S_current_item_indices

        cdef np.ndarray[np.float64_t, ndim=1] item_similarity_np
        cdef np.ndarray[long, ndim=1] relevant_items_partition, relevant_items_partition_sorting, ranking


        if self.structure_similarity:

            S_current_item_indices = self.S_indices[self.S_indptr[current_item]:self.S_indptr[current_item + 1]]

            for item_index in range(len(S_current_item_indices)):

                item_id = S_current_item_indices[item_index]

                Si_mask[item_id] = False

                self.S_data[self.S_indptr[current_item] + item_index] = Si_dense[item_id]
                Si_dense[item_id] = 0.0

        else:

            # structure_dense
            if self.topK == False:

                for item_id in range(self.n_items):
                    Si_mask[item_id] = False
                    self.S_dense[current_item, item_id] = Si_dense[item_id]
                    Si_dense[item_id] = 0.0


            else:

                item_similarity_np = np.zeros(self.n_items, dtype=np.float64)

                for item_id in range(self.n_items):

                    item_similarity_np[item_id] = Si_dense[item_id]

                    Si_mask[item_id] = False
                    Si_dense[item_id] = 0.0


                relevant_items_partition = (-item_similarity_np).argpartition(self.topK)[0:self.topK]
                relevant_items_partition_sorting = np.argsort(-item_similarity_np[relevant_items_partition])
                ranking = relevant_items_partition[relevant_items_partition_sorting]


                # Update global data structure
                start_pos = self.S_indptr[current_item]

                for item_index in range(self.topK):

                    item_id = ranking[item_index]

                    self.S_indices[start_pos + item_index] = item_id
                    self.S_data[start_pos + item_index] = item_similarity_np[item_id]




    cdef double compute_adaptive_gradient(self, int item_id, double gradient):

        cdef double gradient_update

        if self.useAdaGrad:
            self.sgd_cache[item_id] += gradient ** 2

            gradient_update = gradient / (sqrt(self.sgd_cache[item_id]) + 1e-8)


        elif self.useRmsprop:
            self.sgd_cache[item_id] = self.sgd_cache[item_id] * self.gamma + (1 - self.gamma) * gradient ** 2

            gradient_update = gradient / (sqrt(self.sgd_cache[item_id]) + 1e-8)


        elif self.useAdam:

            self.sgd_cache_momentum_1[item_id] = \
                self.sgd_cache_momentum_1[item_id] * self.beta_1 + (1 - self.beta_1) * gradient

            self.sgd_cache_momentum_2[item_id] = \
                self.sgd_cache_momentum_2[item_id] * self.beta_2 + (1 - self.beta_2) * gradient**2


            self.momentum_1 = self.sgd_cache_momentum_1[item_id]/ (1 - self.beta_1_power_t)
            self.momentum_2 = self.sgd_cache_momentum_2[item_id]/ (1 - self.beta_2_power_t)

            gradient_update = self.momentum_1/ (sqrt(self.momentum_2) + 1e-8)

        else:

            gradient_update = gradient


        return gradient_update


    cdef epochIteration_Cython_SGD(self, int n_epochs, int use_BPR, int force_positive, double sample_quota):

        cdef int current_item, current_epoch
        cdef int sample_index, sample_user
        cdef int item_index, item_id
        cdef double sample_rating, gradient_update, gradient, prediction, sample_quota_current_item
        cdef int is_last_sample = False

        cdef int[:] sample_indices, sample_shuffle
        cdef double[:] sample_data

        cdef int[:] user_profile_item_id
        cdef double[:] user_profile_rating

        cdef long start_time = time.time()
        cdef long last_print_time = start_time
        cdef long processed_samples = 0, processed_samples_last_print = 0
        cdef long print_block_size = 100
        cdef long usable_interactions = 0, overall_interactions = 0
        cdef double samples_per_sec, cumulative_loss = 0


        cdef int * Si_mask = < int *> PyMem_Malloc((self.n_items) * sizeof(int))
        cdef double * Si_dense = < double *> PyMem_Malloc((self.n_items) * sizeof(double))



        for current_item in range(self.n_items):

            #print("begin {}".format(time.time() - start_time))

            self.init_S_structure_current_item(current_item, Si_mask, Si_dense)

            #print("initialized {}".format(time.time() - start_time))

            # Get the indices of the users who rated item i
            sample_indices = self.URM_indices_csc[self.URM_indptr_csc[current_item]:self.URM_indptr_csc[current_item + 1]]
            sample_data = self.URM_data_csc[self.URM_indptr_csc[current_item]:self.URM_indptr_csc[current_item + 1]]

            if len(sample_indices) == 0:
                continue



            sample_shuffle = np.arange(0, len(sample_indices), dtype=np.int32)


            if self.useAdaGrad:
                self.sgd_cache = np.zeros((self.n_items), dtype=np.float64)

            elif self.useRmsprop:
                self.sgd_cache = np.zeros((self.n_items), dtype=np.float64)

            elif self.useAdam:
                self.sgd_cache_momentum_1 = np.zeros((self.n_items), dtype=np.float64)
                self.sgd_cache_momentum_2 = np.zeros((self.n_items), dtype=np.float64)

                self.momentum_1 = 0.0
                self.momentum_2 = 0.0

                self.beta_1_power_t = self.beta_1
                self.beta_2_power_t = self.beta_2


            # Adjust sample percentage for current item to ensure at least one is chosen
            sample_quota_current_item = 1/len(sample_indices)

            if sample_quota_current_item < sample_quota:
                sample_quota_current_item = sample_quota



            for current_epoch in range(n_epochs):

                np.random.shuffle(sample_shuffle)

                #print("sample_shuffle {}".format(time.time() - start_time))

                for sample_index in range(len(sample_indices)):

                    if rand() > RAND_MAX * sample_quota:
                        continue

                    sample_user = sample_indices[sample_shuffle[sample_index]]
                    sample_rating = sample_data[sample_shuffle[sample_index]]

                    user_profile_item_id = self.URM_indices[self.URM_indptr[sample_user]:self.URM_indptr[sample_user + 1]]
                    user_profile_rating = self.URM_data[self.URM_indptr[sample_user]:self.URM_indptr[sample_user + 1]]

                    #print("user_profile {}".format(time.time() - start_time))

                    if use_BPR:

                        prediction = 0.0

                        for item_index in range(len(user_profile_item_id)):
                            item_id = user_profile_item_id[item_index]

                            overall_interactions += 1

                            if Si_mask[item_id] == True and item_id != current_item:
                                prediction += 1.0 * Si_dense[item_id]
                                usable_interactions += 1

                        gradient = 1 / (1 + exp(prediction))

                        cumulative_loss += prediction**2


                    else:   ## USE MSE

                        #prediction = + self.user_bias[sample_user]
                        prediction = 0.0

                        for item_index in range(len(user_profile_item_id)):
                            item_id = user_profile_item_id[item_index]

                            overall_interactions += 1

                            if Si_mask[item_id] == True and item_id != current_item:
                                prediction += user_profile_rating[item_index] * Si_dense[item_id]
                                usable_interactions += 1


                        gradient = prediction - sample_rating

                        cumulative_loss += gradient**2



                    #print("gradient {}".format(time.time() - start_time))




                    for item_index in range(len(user_profile_item_id)):
                        item_id = user_profile_item_id[item_index]

                        if Si_mask[item_id] == True and item_id != current_item:

                            gradient_update = self.compute_adaptive_gradient(item_id, gradient)

                            if use_BPR:
                                Si_dense[item_id] += self.learning_rate * (gradient_update - Si_dense[item_id] * self.lambda_2 - self.lambda_1)
                            else:
                                Si_dense[item_id] -=  self.learning_rate * (gradient_update * user_profile_rating[item_index] + Si_dense[item_id] * self.lambda_2 + self.lambda_1)


                            if force_positive and Si_dense[item_id] < 0.0:
                                Si_dense[item_id] = 0.0


                    #print("gradient_update {}".format(time.time() - start_time))
                    #input()



                    processed_samples += 1
                    processed_samples_last_print +=1

                    # Exponentiation of beta at the end of each sample
                    if self.useAdam:

                        self.beta_1_power_t *= self.beta_1
                        self.beta_2_power_t *= self.beta_2


                # END for sample_indices

                is_last_sample = (current_item == self.n_items -1) and \
                                 (current_epoch == n_epochs -1) and \
                                 ((sample_index == len(sample_indices) -1 ) or len(sample_indices)==0)


                if processed_samples_last_print >= print_block_size or is_last_sample:

                    current_time = time.time()

                    # Set block size to the number of items necessary in order to print every 30 seconds
                    samples_per_sec = processed_samples/(time.time()-start_time)

                    print_block_size = int(samples_per_sec*30)

                    if current_time - last_print_time > 30  or is_last_sample:

                        print("Processed {:.2E} samples ( {:2.0f} % ), {:.2E} samples/sec. Average loss is {:.2E}. Usable interactions are {:.2E} ( {:2.0f} % ). Elapsed time {:.2f} min".format(
                            processed_samples, processed_samples*1.0/(len(self.URM_data)*n_epochs)*100,
                            samples_per_sec,
                            cumulative_loss/processed_samples,
                            usable_interactions,
                            usable_interactions*1.0/overall_interactions*100,
                            (time.time()-start_time) / 60))

                        last_print_time = current_time
                        processed_samples_last_print = 0

                        sys.stdout.flush()
                        sys.stderr.flush()



            #print("End for epochs {}".format(time.time() - start_time))

            # End for epochs

            self.clear_and_save_S_structure_current_item(current_item, Si_mask, Si_dense)

            #print("clear_and_save_S_structure_current_item {}".format(time.time() - start_time))
            #input()


        PyMem_Free(Si_mask)
        PyMem_Free(Si_dense)


        print("SLIM_Structure, fit complete!")


    def epochIteration_Cython_batch(self, epochs=30):


        cdef int current_item, current_epoch, n_epochs = epochs
        cdef int sample_index, sample_user, sample_index_start_batch, sample_index_end_batch, samples_completed
        cdef int item_index, item_id
        cdef double prediction, error, sample_rating, gradient_update, gradient
        cdef int is_last_sample = False

        cdef int[:] sample_indices, sample_shuffle
        cdef double[:] sample_data

        cdef int[:] user_profile_item_id
        cdef double[:] user_profile_rating
        cdef int[:] user_id_in_batch = np.zeros(self.n_users, dtype = np.int32)

        cdef long start_time = time.time()
        cdef long last_print_time = start_time
        cdef long processed_samples = 0, print_block_size = 100000
        cdef double samples_per_sec, cumulative_loss = 0


        cdef int * Si_mask = < int *> PyMem_Malloc((self.n_items) * sizeof(int))
        cdef double * Si_dense = < double *> PyMem_Malloc((self.n_items) * sizeof(double))



        for current_item in range(self.n_items):

            self.init_S_structure_current_item(current_item, Si_mask, Si_dense)

            # Get the indices of the users who rated item i
            sample_indices = self.URM_indices_csc[self.URM_indptr_csc[current_item]:self.URM_indptr_csc[current_item + 1]]
            sample_data = self.URM_data_csc[self.URM_indptr_csc[current_item]:self.URM_indptr_csc[current_item + 1]]

            sample_shuffle = np.arange(0, len(sample_indices), dtype=np.int32)


            if self.useAdaGrad:
                self.sgd_cache = np.zeros((self.n_items), dtype=np.float64)

            elif self.useRmsprop:
                self.sgd_cache = np.zeros((self.n_items), dtype=np.float64)

            elif self.useAdam:
                self.sgd_cache_momentum_1 = np.zeros((self.n_items), dtype=np.float64)
                self.sgd_cache_momentum_2 = np.zeros((self.n_items), dtype=np.float64)

                self.momentum_1 = 0.0
                self.momentum_2 = 0.0

                self.beta_1_power_t = self.beta_1
                self.beta_2_power_t = self.beta_2



            for current_epoch in range(n_epochs):

                np.random.shuffle(sample_shuffle)

                sample_index_start_batch = 0
                sample_index_end_batch = 0
                samples_completed = False

                while not samples_completed:

                    # Set new batch boundaries
                    sample_index_start_batch = sample_index_end_batch
                    sample_index_end_batch += self.batch_size

                    if sample_index_end_batch >= len(sample_indices):
                        samples_completed = True
                        sample_index_end_batch = len(sample_indices) -1

                    error = 0.0

                    #print("Batch {}-{}".format(sample_index_start_batch, sample_index_end_batch))

                    # Process samples in batch
                    for sample_index in range(sample_index_start_batch, sample_index_end_batch):

                        sample_user = sample_indices[sample_shuffle[sample_index]]
                        sample_rating = sample_data[sample_shuffle[sample_index]]

                        user_id_in_batch[sample_user] = True

                        user_profile_item_id = self.URM_indices[self.URM_indptr[sample_user]:self.URM_indptr[sample_user + 1]]
                        user_profile_rating = self.URM_data[self.URM_indptr[sample_user]:self.URM_indptr[sample_user + 1]]


                        prediction = - self.user_bias[sample_user]

                        for item_index in range(len(user_profile_item_id)):
                            item_id = user_profile_item_id[item_index]

                            if Si_mask[item_id] == True and item_id != current_item:
                                prediction += user_profile_rating[item_index] * Si_dense[item_id]



                        error += prediction - sample_rating


                    # Compute average cumulative loss and average gradient
                    error /= self.batch_size
                    cumulative_loss += (error/self.batch_size)**2 * self.batch_size



                    # Update
                    for sample_index in range(sample_index_start_batch, sample_index_end_batch):

                        sample_user = sample_indices[sample_shuffle[sample_index]]

                        processed_samples += 1

                        # Update each user only one time
                        if user_id_in_batch[sample_user]:
                            user_id_in_batch[sample_user] = False

                            user_profile_item_id = self.URM_indices[self.URM_indptr[sample_user]:self.URM_indptr[sample_user + 1]]
                            user_profile_rating = self.URM_data[self.URM_indptr[sample_user]:self.URM_indptr[sample_user + 1]]

                            for item_index in range(len(user_profile_item_id)):
                                item_id = user_profile_item_id[item_index]

                                if Si_mask[item_id] == True and item_id != current_item:

                                    gradient_update = self.compute_adaptive_gradient(item_id, error)

                                    Si_dense[item_id] -= gradient_update * user_profile_rating[item_index] * self.learning_rate \
                                                         + Si_dense[item_id] * self.lambda_2 + self.lambda_1

                                    if Si_dense[item_id] < 0.0:
                                        Si_dense[item_id] = 0.0







                    is_last_sample = (current_item == self.n_items -1) and \
                                        samples_completed and \
                                        (current_epoch == n_epochs -1)

                    if processed_samples % print_block_size==0 or is_last_sample:

                        current_time = time.time()

                        # Set block size to the number of items necessary in order to print every 30 seconds
                        samples_per_sec = processed_samples/(time.time()-start_time)

                        print_block_size = int(samples_per_sec*30)

                        if current_time - last_print_time > 30  or is_last_sample:

                            print("Processed {:.2E} samples ( {:2.0f} % ), {:.2E} samples/sec. Average loss is {:.2E}. Elapsed time {:.2f} min".format(
                                processed_samples, processed_samples*1.0/(len(self.URM_data)*n_epochs)*100,
                                samples_per_sec,
                                cumulative_loss/processed_samples,
                                (time.time()-start_time) / 60))

                            last_print_time = current_time

                            sys.stdout.flush()
                            sys.stderr.flush()



                    # Exponentiation of beta at the end of each batch
                    if self.useAdam:

                        self.beta_1_power_t *= self.beta_1
                        self.beta_2_power_t *= self.beta_2


                #End for samples
            # End for epochs

            self.clear_and_save_S_structure_current_item(current_item, Si_mask, Si_dense)


        PyMem_Free(Si_mask)
        PyMem_Free(Si_dense)


        print("SLIM_Structure, fit complete!")





    def get_S(self):

        if self.structure_full and self.topK == False:
            return np.array(self.S_dense)

        S_sparse = sps.csr_matrix((np.array(self.S_data), np.array(self.S_indices), np.array(self.S_indptr)), shape=(self.n_items, self.n_items), dtype=np.float32)


        return S_sparse
