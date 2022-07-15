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

from scipy import sparse as sps
from Base.Cython.cosine_similarity import Cosine_Similarity
from Recommenders.Recommender_utils import check_matrix

import time, sys

cdef class SLIM_Structure:

    cdef int[:] S_indices
    cdef int[:] S_indptr
    cdef float[:] S_data

    cdef int[:] URM_indices, URM_indptr
    cdef float[:] URM_data

    cdef int[:] URM_indices_csc, URM_indptr_csc
    cdef float[:] URM_data_csc

    cdef double learning_rate, b, g

    cdef int n_items, n_users


    def __init__(self, S_structure, URM, learning_rate=0.001, b=0.001, g=0.001):

        self.learning_rate = learning_rate
        self.b = b
        self.g = g

        S_structure = check_matrix(S_structure, "csr")
        self.S_indices = np.array(S_structure.indices, dtype=np.int32)
        self.S_indptr = np.array(S_structure.indptr, dtype=np.int32)
        self.S_data = np.array(S_structure.data, dtype=np.float64)


        URM = check_matrix(URM, "csr")
        self.URM_indices = np.array(URM.indices, dtype=np.int32)
        self.URM_indptr = np.array(URM.indptr, dtype=np.int32)
        self.URM_data = np.array(URM.data, dtype=np.float64)

        URM = check_matrix(URM, "csc")
        self.URM_indices_csc = np.array(URM.indices, dtype=np.int32)
        self.URM_indptr_csc = np.array(URM.indptr, dtype=np.int32)
        self.URM_data_csc = np.array(URM.data, dtype=np.float64)


        self.n_users = URM.shape[0]
        self.n_items = URM.shape[1]





    cdef init_S_structure_current_item(self, int current_item, int[:] *Si_mask, double[:] *Si_dense)

        cdef int item_index, item_id

        cdef int[:] S_current_item_indices = self.S_indices[self.S_indptr[current_item]:self.S_indptr[current_item + 1]]
        cdef double[:] S_current_item_data = self.S_data[self.S_indptr[current_item]:self.S_indptr[current_item + 1]]


        for item_index in range(len(S_current_item_indices)):

            item_id = S_current_item_indices[item_index]

            *Si_mask[item_id] = True
            *Si_dense[item_id] = S_current_item_data[item_index]



    cdef clear_and_save_S_structure_current_item(self, int current_item, int[:] *Si_mask, double[:] *Si_dense)

        cdef int item_index, item_id

        cdef int[:] S_current_item_indices = self.S_indices[self.S_indptr[current_item]:self.S_indptr[current_item + 1]]

        for item_index in range(len(S_current_item_indices)):

            item_id = S_current_item_indices[item_index]

            *Si_mask[item_id] = False

            self.S_data[self.S_indptr[current_item] + item_index] = *Si_dense[item_id]
            *Si_dense[item_id] =0.0






    def fit(self, epochs):


        cdef int[:] Si_mask = np.zeros(items, np.int32)
        cdef double[:] Si_dense = np.zeros(items, np.float32)

        cdef int current_item, current_epoch, epochs = epochs
        cdef int sample_index, sample_user, sample_rating
        cdef int item_index, item_id
        cdef double prediction, error

        cdef int[:] sample_indices
        cdef double[:] sample_data

        cdef int[:] user_profile_item_id
        cdef double[:] user_profile_rating

        cdef long start_time = time.time()
        cdef long last_print_time = start_time
        cdef long processed_samples = 0, print_block_size = 100000
        cdef double samples_per_sec



        for current_item in range(self.n_items):

            self.init_S_structure_current_item(current_item, &Si_mask, &Si_dense)

            # Get the indices of the users who rated item i
            sample_indices = self.URM_indices_csc[self.URM_indptr_csc[current_item]:self.URM_indptr_csc[current_item + 1]]
            sample_data = self.URM_data_csc[self.URM_indptr_csc[current_item]:self.URM_indptr_csc[current_item + 1]]


            for current_epoch in range(epochs):

                for sample_index in range(len(sample_indices)):

                    sample_user = sample_indices[sample_index]
                    sample_rating = sample_data[sample_index]

                    user_profile_item_id = self.URM_indices[self.URM_indptr[sample_user]:self.URM_indptr[sample_user + 1]]
                    user_profile_rating = self.URM_data[self.URM_indptr[sample_user]:self.URM_indptr[sample_user + 1]]


                    prediction = 0.0

                    for item_index in range(len(user_profile_item_id)):
                        item_id = user_profile_item_id[item_index]

                        if Si_mask[item_id] == True and item_id != current_item:
                            prediction += user_profile_rating[item_index] * Si_dense[item_id]



                    error = prediction - sample_rating


                    for item_index in range(len(user_profile_item_id)):
                        item_id = user_profile_item_id[item_index]

                        if Si_mask[item_id] == True and item_id != current_item:
                            Si_dense[item_id] -= error * user_profile_rating[item_index] * self.learning_rate + Si_dense[item_id] * self.b + self.g



                    processed_samples += 1

                    if processed_samples % print_block_size==0 or current_item == self.n_items -1:

                        current_time = time.time()

                        # Set block size to the number of items necessary in order to print every 30 seconds
                        samples_per_sec = processed_samples/(time.time()-start_time)

                        print_block_size = int(samples_per_sec*30)

                        if current_time - last_print_time > 30  or current_item == self.n_items -1:

                            print("Processed {} samples ( {:2.0f} % ), {:.2f} samples/sec, elapsed time {:.2f} min".format(
                                processed_samples, processed_samples*1.0/(len(self.URM_data)*epochs)*100, samples_per_sec, (time.time()-start_time) / 60))

                            last_print_time = current_time

                            sys.stdout.flush()
                            sys.stderr.flush()


                #End for samples
            # End for epochs

            self.clear_and_save_S_structure_current_item(self, int current_item, int[:] *Si_mask, double[:] *Si_dense)


        print("SLIM_Structure, fit complete!")


