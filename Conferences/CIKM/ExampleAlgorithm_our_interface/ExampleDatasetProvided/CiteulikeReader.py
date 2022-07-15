#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/11/18

@author: Maurizio Ferrari Dacrema
"""


from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

import scipy.io
import scipy.sparse as sps
import h5py, os
import numpy as np

from Recommenders.DataIO import DataIO
from Recommenders.Recommender_utils import reshapeSparse

class CiteulikeReader(object):

    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, pre_splitted_path):

        super(CiteulikeReader, self).__init__()

        pre_splitted_path += "data_split/"
        pre_splitted_filename = "splitted_data_"

        # TODO Replace this with the relative path of the data files provided by the authors, if present
        original_data_path = os.path.join(os.path.dirname(__file__), '..' , ".." , "ExampleAlgorithm_github/data/citeulike/")

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        dataIO = DataIO(pre_splitted_path)

        try:

            print("CiteulikeReader: Attempting to load pre-splitted data")

            for attrib_name, attrib_object in dataIO.load_data(pre_splitted_filename).items():
                 self.__setattr__(attrib_name, attrib_object)


        except FileNotFoundError:

            print("CiteulikeReader: Pre-splitted data not found, building new one")

            print("CiteulikeReader: loading URM")

            # TODO Replace this with the code required to load the data, as written in the original source code
            # TODO Then transform whatever data structure it originally had into a sparse matrix
            # TODO If the original training-validation-test is provided, load the provided files
            # TODO if only the training-test split is provided, create a validation by splitting the training data with the same split strategy used for the test

            URM_train_builder = self._load_data_file(original_data_path + "cf-train-users.dat")
            URM_test_builder = self._load_data_file(original_data_path + "cf-test-users.dat")

            URM_test = URM_test_builder.get_SparseMatrix()
            URM_train = URM_train_builder.get_SparseMatrix()

            ICM_tokens_TFIDF = scipy.io.loadmat(original_data_path + "mult_nor.mat")['X']
            ICM_tokens_TFIDF = sps.csr_matrix(ICM_tokens_TFIDF)

            ICM_tokens_bool = ICM_tokens_TFIDF.copy()
            ICM_tokens_bool.data = np.ones_like(ICM_tokens_bool.data)


            n_rows = max(URM_test.shape[0], URM_train.shape[0])
            n_cols = max(URM_test.shape[1], URM_train.shape[1], ICM_tokens_TFIDF.shape[0])

            newShape = (n_rows, n_cols)

            URM_test = reshapeSparse(URM_test, newShape)
            URM_train = reshapeSparse(URM_train, newShape)

            # Split the train data in train and validation
            URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train.copy(), train_percentage = 0.8)


            # TODO get the sparse matrices in the correct dictionary with the correct name
            # TODO ICM_DICT and UCM_DICT can be empty if no ICMs or UCMs are required
            self.ICM_DICT = {
                "ICM_tokens_TFIDF": ICM_tokens_TFIDF,
                "ICM_tokens_bool": ICM_tokens_bool,
            }

            self.UCM_DICT = {}

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_test": URM_test,
                "URM_validation": URM_validation,
            }


            # You likely will not need to modify this part
            data_dict_to_save = {
                "ICM_DICT": self.ICM_DICT,
                "UCM_DICT": self.UCM_DICT,
                "URM_DICT": self.URM_DICT,
            }

            dataIO.save_data(pre_splitted_filename, data_dict_to_save=data_dict_to_save)

            print("CiteulikeReader: loading complete")







    def _load_data_file(self, filePath, separator = " "):

        URM_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, auto_create_col_mapper=False)

        fileHandle = open(filePath, "r")
        user_index = 0


        for line in fileHandle:

            if (user_index % 1000000 == 0):
                print("Processed {} cells".format(user_index))

            if (len(line)) > 1:

                line = line.replace("\n", "")
                line = line.split(separator)

                if len(line)>0:

                    if line[0]!="0":

                        line = [int(line[i]) for i in range(len(line))]

                        URM_builder.add_single_row(user_index, line[1:], data=1.0)

            user_index += 1


        fileHandle.close()

        return  URM_builder