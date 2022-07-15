#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/12/18

@author: Emanuele Chioso, Maurizio Ferrari Dacrema
"""

from skopt import gp_minimize
import pandas as pd
import numpy as np
import time, os
from skopt.space import Real, Integer, Categorical
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
from Recommenders.DataIO import DataIO
from HyperparameterTuning.SearchAbstractClass import SearchAbstractClass
import traceback


class SearchCleanTrailingExceptionsSkopt(SearchAbstractClass):

    ALGORITHM_NAME = None


    def _resume_from_saved(self):

        try:
            self.metadata_dict = self.dataIO.load_data(file_name = self.output_file_name_root + "_metadata")
            self.ALGORITHM_NAME = self.metadata_dict['algorithm_name_search']

            # This code can be used to remove all explored cases starting from the first exception raised
            # Useful if, for example, you accidentally saturate the RAM and get memory errors, and want to clean
            # the metadata to continue the search from before the exceptions were raised
            start = None
            for i, exc in enumerate(self.metadata_dict["exception_list"]):
                if exc is not None and start is None:
                    start=i

            if start is not None:
                self._print("{}: '{}'... Removing starting from case {}.".format(self.ALGORITHM_NAME, self.output_file_name_root, start))
                self._remove_intermediate_cases(list(range(start, len(self.metadata_dict['hyperparameters_df']))))

        except FileNotFoundError:
            self._write_log("{}: Resuming '{}' Failed, no such file exists.\n".format(self.ALGORITHM_NAME, self.output_file_name_root))
            self.resume_from_saved = False
            return None, None

        except Exception as e:
            self._write_log("{}: Resuming '{}' Failed, generic exception: {}.\n".format(self.ALGORITHM_NAME, self.output_file_name_root, str(e)))
            raise e

        raise KeyboardInterrupt



    def search(self,
               # recommender_input_args,
               # hyperparameter_search_space,
               metric_to_optimize,
               cutoff_to_optimize,
               # n_cases = None,
               # n_random_starts = None,
               output_folder_path,
               output_file_name_root,
               save_model = "best",
               # save_metadata = True,
               # resume_from_saved = False,
               # recommender_input_args_last_test = None,
               evaluate_on_test = "best",
               # max_total_time = None
               ):

        self.evaluate_on_test = evaluate_on_test
        self.cutoff_to_optimize = cutoff_to_optimize
        self.metric_to_optimize = metric_to_optimize
        self.output_file_name_root = output_file_name_root
        self.output_folder_path = output_folder_path
        self.save_model = save_model
        self.dataIO = DataIO(folder_path = output_folder_path)

        self._resume_from_saved()


