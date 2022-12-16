#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 04/06/22

@author: Zhitao He
"""

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix

import scipy.io
import scipy.sparse as sps
import h5py, os
import numpy as np
import pandas as pd

from Recommenders.DataIO import DataIO
from Recommenders.Recommender_utils import reshapeSparse


class MultiDatasetsReader(object):
    URM_DICT = {}
    ICM_DICT = {}

    def __init__(self, path):

        super(MultiDatasetsReader, self).__init__()

        pre_splitted_path = path + "/data_split/"
        pre_splitted_filename = "splitted_data_"

        # If directory does not exist, create
        if not os.path.exists(pre_splitted_path):
            os.makedirs(pre_splitted_path)

        dataIO = DataIO(pre_splitted_path)

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        try:
            print("MultiDatasetsReader: Pre-splitted data found")

            print("MultiDatasetsReader: loading URM")

            # TODO Replace this with the code required to load the data, as written in the original source code
            # TODO Then transform whatever data structure it originally had into a sparse matrix
            # TODO If the original training-validation-test is provided, load the provided files
            # TODO if only the training-test split is provided, create a validation by splitting the training data with the same split strategy used for the test

            URM_train_builder = self._load_data_file(train_file)
            URM_test_builder = self._load_data_file(test_file)

            URM_test = URM_test_builder.get_SparseMatrix()
            URM_train = URM_train_builder.get_SparseMatrix()

            n_rows = max(URM_test.shape[0], URM_train.shape[0])
            n_cols = max(URM_test.shape[1], URM_train.shape[1])

            print(
                "The percentage of training set: %.1f, test set: %.1f" % (URM_train.nnz / (URM_train.nnz + URM_test.nnz)
                                                                          ,
                                                                          URM_test.nnz / (URM_train.nnz + URM_test.nnz))
            )
            print("#interactions: %d" % (URM_train.nnz + URM_test.nnz))
            print("Maximum number of row: %d, column: %d" % (n_rows, n_cols))

            newShape = (n_rows, n_cols)

            URM_test = reshapeSparse(URM_test, newShape)
            URM_train = reshapeSparse(URM_train, newShape)

            # Split the train data in train and validation
            URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train.copy(),
                                                                                    train_percentage=0.8)

            print("#interactions of URM_train: %d, URM_validation: %d, URM_test: %d" % (
                URM_train.nnz, URM_validation.nnz, URM_test.nnz))

            # TODO get the sparse matrices in the correct dictionary with the correct name
            # TODO ICM_DICT and UCM_DICT can be empty if no ICMs or UCMs are required
            self.ICM_DICT = {}

            self.UCM_DICT = {}

            self.URM_DICT = {
                "URM_train": URM_train,
                "URM_validation": URM_validation,
                "URM_test": URM_test,
            }

            # You likely will not need to modify this part
            data_dict_to_save = {
                "ICM_DICT": self.ICM_DICT,
                "UCM_DICT": self.UCM_DICT,
                "URM_DICT": self.URM_DICT,
            }

            dataIO.save_data(pre_splitted_filename, data_dict_to_save=data_dict_to_save)

            print("MultiDatasetsReader: loading complete")

        except Exception as e:
            print(e)

    def _load_data_file(self, filePath, separator=" "):

        URM_builder = IncrementalSparseMatrix(auto_create_row_mapper=False, auto_create_col_mapper=False)

        fileHandle = open(filePath, "r").readlines()

        for line in fileHandle:
            tmps = line.strip()
            inters = [int(i) for i in tmps.split(' ')]

            u_id, pos_ids = inters[0], inters[1:]
            pos_ids = list(set(pos_ids))

            URM_builder.add_single_row(u_id, pos_ids, data=1.0)

        return URM_builder
