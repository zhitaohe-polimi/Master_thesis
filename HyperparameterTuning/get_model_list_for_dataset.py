#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/05/2022

@author: Maurizio Ferrari Dacrema
"""

import numpy as np
import os, traceback, multiprocessing
from argparse import ArgumentParser
from functools import partial

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.Recommender_import_list import *
from Data_manager import *

# from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
# from HyperparameterTuning.SearchSingleCase import SearchSingleCase
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative, runHyperparameterSearch_Content, runHyperparameterSearch_Hybrid

from Data_manager.data_consistency_check import assert_implicit_data, assert_disjoint_matrices

from Utils.plot_popularity import plot_popularity_bias, save_popularity_statistics
# from Utils.ResultFolderLoader import ResultFolderLoader, generate_latex_hyperparameters
from Utils.all_dataset_stats_latex_table import all_dataset_stats_latex_table
from Data_manager.DataSplitter_Holdout import DataSplitter_Holdout
from Data_manager.DataSplitter_leave_k_out import DataSplitter_leave_k_out
from Utils.RecommenderInstanceIterator import RecommenderInstanceIterator


from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender

def _get_model_list_for_dataset(recommender_class_list, KNN_similarity_list, ICM_dict, UCM_dict):

    recommender_class_list = recommender_class_list.copy()

    # Model list format: recommender class, KNN heuristic, ICM/UCM name, ICM/UCM matrix
    model_list = []

    for recommender_class in recommender_class_list:

        if issubclass(recommender_class, BaseItemCBFRecommender):
            for ICM_name, ICM_object in ICM_dict.items():
                if recommender_class in [ItemKNNCBFRecommender, ItemKNN_CFCBF_Hybrid_Recommender]:
                    for KNN_similarity in KNN_similarity_list:
                        model_list.append((recommender_class, KNN_similarity, ICM_name, ICM_object))
                else:
                    model_list.append((recommender_class, None, ICM_name, ICM_object))

        elif issubclass(recommender_class, BaseUserCBFRecommender):
            for UCM_name, UCM_object in UCM_dict.items():
                if recommender_class in [UserKNNCBFRecommender, UserKNN_CFCBF_Hybrid_Recommender]:
                    for KNN_similarity in KNN_similarity_list:
                        model_list.append((recommender_class, KNN_similarity, UCM_name, UCM_object))
                else:
                    model_list.append((recommender_class, None, UCM_name, UCM_object))


        else:
            if recommender_class in [ItemKNNCFRecommender, UserKNNCFRecommender]:
                for KNN_similarity in KNN_similarity_list:
                    model_list.append((recommender_class, KNN_similarity, None, None))

            else:
                model_list.append((recommender_class, None, None, None))

    return model_list


def _optimize_single_model(model_tuple, URM_train, URM_train_last_test = None,
                          n_cases = None, n_random_starts = None, resume_from_saved = False,
                          save_model = "best", evaluate_on_test = "best", max_total_time = None,
                          evaluator_validation = None, evaluator_test = None, evaluator_validation_earlystopping = None,
                          metric_to_optimize = None, cutoff_to_optimize = None,
                          model_folder_path ="result_experiments/"):

    try:

        recommender_class, KNN_similarity, ICM_UCM_name, ICM_UCM_object = model_tuple

        if recommender_class in [ItemKNN_CFCBF_Hybrid_Recommender, UserKNN_CFCBF_Hybrid_Recommender,
                                 LightFMUserHybridRecommender, LightFMItemHybridRecommender]:
            runHyperparameterSearch_Hybrid(recommender_class,
                                           URM_train=URM_train,
                                           URM_train_last_test=URM_train_last_test,
                                           metric_to_optimize=metric_to_optimize,
                                           cutoff_to_optimize=cutoff_to_optimize,
                                           evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                           evaluator_validation=evaluator_validation,
                                           similarity_type_list=[KNN_similarity],
                                           evaluator_test=evaluator_test,
                                           max_total_time=max_total_time,
                                           output_folder_path=model_folder_path,
                                           parallelizeKNN=False,
                                           allow_weighting=True,
                                           save_model = save_model,
                                           evaluate_on_test = evaluate_on_test,
                                           resume_from_saved=resume_from_saved,
                                           ICM_name=ICM_UCM_name,
                                           ICM_object=ICM_UCM_object.copy(),
                                           n_cases=n_cases,
                                           n_random_starts=n_random_starts)

        elif issubclass(recommender_class, BaseItemCBFRecommender) or issubclass(recommender_class, BaseUserCBFRecommender):
            runHyperparameterSearch_Content(recommender_class,
                                            URM_train=URM_train,
                                            URM_train_last_test=URM_train_last_test,
                                            metric_to_optimize=metric_to_optimize,
                                            cutoff_to_optimize=cutoff_to_optimize,
                                            evaluator_validation=evaluator_validation,
                                            similarity_type_list=[KNN_similarity],
                                            evaluator_test=evaluator_test,
                                            output_folder_path=model_folder_path,
                                            parallelizeKNN=False,
                                            allow_weighting=True,
                                            save_model = save_model,
                                            evaluate_on_test = evaluate_on_test,
                                            max_total_time=max_total_time,
                                            resume_from_saved=resume_from_saved,
                                            ICM_name=ICM_UCM_name,
                                            ICM_object=ICM_UCM_object.copy(),
                                            n_cases=n_cases,
                                            n_random_starts=n_random_starts)

        else:

            runHyperparameterSearch_Collaborative(recommender_class, URM_train=URM_train,
                                           URM_train_last_test=URM_train_last_test,
                                           metric_to_optimize=metric_to_optimize,
                                           cutoff_to_optimize=cutoff_to_optimize,
                                           evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                           evaluator_validation=evaluator_validation,
                                           similarity_type_list = [KNN_similarity],
                                           evaluator_test=evaluator_test,
                                           max_total_time=max_total_time,
                                           output_folder_path=model_folder_path,
                                           resume_from_saved=resume_from_saved,
                                           parallelizeKNN=False,
                                           allow_weighting=True,
                                           save_model = save_model,
                                           evaluate_on_test = evaluate_on_test,
                                           n_cases=n_cases,
                                           n_random_starts=n_random_starts)


    except Exception as e:
        print("On CBF recommender {} Exception {}".format(model_tuple[0], str(e)))
        traceback.print_exc()




