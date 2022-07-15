#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Recommender_import_list import *

from ParameterTuning.run_parameter_search import runParameterSearch_Collaborative, runParameterSearch_Content, runParameterSearch_Hybrid
from ParameterTuning.SearchSingleCase import SearchSingleCase
from ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs

from Utils.print_results_latex_table import print_time_statistics_latex_table, print_results_latex_table, print_hyperparameters_latex_table

from functools import partial
import numpy as np
import pandas as pd
import os, traceback, argparse

from Conferences.SIGIR.CARA_our_interface.Brightkite.BrightkiteReader import BrightkiteReader
from Conferences.SIGIR.CARA_our_interface.CARA_RecommenderWrapper import CARA_RecommenderWrapper
from Conferences.SIGIR.CARA_our_interface.EvaluatorContextWrapper import EvaluatorContextWrapper

from Base.Evaluation.Evaluator import EvaluatorHoldout, EvaluatorNegativeItemSample
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices


def read_data_split_and_search(dataset_name,
                                   flag_baselines_tune=False,
                                   flag_DL_article_default=False, flag_DL_tune=False,
                                   flag_print_results=False):


    np.random.seed(1)

    result_folder_path = "result_experiments/{}/{}_{}/".format(CONFERENCE_NAME, ALGORITHM_NAME, dataset_name)

    pre_splitted_path = "Data_manager_split_datasets/Brightkite/SIGIR/CARA_our_interface/"

    perform_also_split_disjoint = True

    if dataset_name == "brightkite":
        dataset = BrightkiteReader(result_folder_path, split_validation=True, split_disjoint=perform_also_split_disjoint)
        dataset_last_test = BrightkiteReader(result_folder_path, split_validation=False, split_disjoint=perform_also_split_disjoint)
    else:
        print("Dataset name not supported, current is {}".format(dataset_name))
        return


    print ('Current dataset is: {}'.format(dataset_name))

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_validation_negative = dataset.URM_DICT["URM_validation_negative"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()
    URM_test_negative = dataset.URM_DICT["URM_test_negative"].copy()

    df_test = dataset.URM_DICT["dataframe_test"]
    df_validation = dataset.URM_DICT["dataframe_validation"]
    df_train = dataset.URM_DICT["dataframe_train"]

    assert_implicit_data([URM_train, URM_test, URM_test_negative])
    assert_disjoint_matrices([URM_test_negative, URM_test])
    assert_disjoint_matrices([URM_test_negative, URM_validation])
    assert_disjoint_matrices([URM_test_negative, URM_train])
    assert_disjoint_matrices([URM_validation_negative, URM_validation])
    assert_disjoint_matrices([URM_validation_negative, URM_train])

    URM_train_last_test = dataset_last_test.URM_DICT["URM_train"]
    dataframe_train_last_test = dataset_last_test.URM_DICT["dataframe_train"]

    # If directory does not exist, create
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    # TODO Replace metric to optimize and cutoffs
    metric_to_optimize = "NDCG"
    cutoff_list_validation = [10]
    cutoff_list_test = [10]

    n_cases = 50
    n_random_starts = 15

    cold_user_ids = []
    normal_user_ids = []
    for uid, group in df_test.groupby('uid'):
      checkin_count = df_train[df_train.uid == uid].shape[0]
      if checkin_count >= 10:
        normal_user_ids.append(uid - 1)
      else:
        cold_user_ids.append(uid - 1)

    # change ignore_users to switch evaluation from cold to normal users
    cold_evaluation = False
    if cold_evaluation:
        ignore_users = cold_user_ids
    else:
        ignore_users = normal_user_ids

    evaluator_test = EvaluatorNegativeItemSample(URM_test, URM_test_negative, cutoff_list=cutoff_list_test, exclude_seen=False, ignore_users=ignore_users)
    evaluator_test_context = EvaluatorContextWrapper(evaluator_test, df_test)

    evaluator_validation = EvaluatorNegativeItemSample(URM_validation, URM_validation_negative, cutoff_list=cutoff_list_test, exclude_seen=False, ignore_users=ignore_users)
    evaluator_validation_context = EvaluatorContextWrapper(evaluator_validation, df_validation)

    ################################################################################################
    ######
    ######      DL ALGORITHM
    ######

    if flag_DL_article_default:

        # TODO fill this dictionary with the hyperparameters of the algorithm
        article_hyperparameters = {
                              "batch_size": 256,
                              "epochs": 2,
                              "latent_dim": 10,
                              "max_venue": 5
                              }

        # Do not modify earlystopping
        earlystopping_hyperparameters = {"validation_every_n": 1, #TODO CHANGE THIS
                                    "stop_on_validation": True,
                                    "lower_validations_allowed": 5,
                                    "evaluator_object": evaluator_validation_context,
                                    "validation_metric": metric_to_optimize,
                                    }


        # Fit the DL model, select the optimal number of epochs and save the result
        parameterSearch = SearchSingleCase(CARA_RecommenderWrapper,
                                           evaluator_validation=evaluator_validation_context,
                                           evaluator_test=evaluator_test_context)

        recommender_input_args = SearchInputRecommenderArgs(
                                            CONSTRUCTOR_POSITIONAL_ARGS = [URM_train, df_train],
                                            FIT_KEYWORD_ARGS = earlystopping_hyperparameters)

        recommender_input_args_last_test = recommender_input_args.copy()
        recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train_last_test
        recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[1] = dataframe_train_last_test

        parameterSearch.search(recommender_input_args,
                               recommender_input_args_last_test = recommender_input_args_last_test,
                               fit_hyperparameters_values=article_hyperparameters,
                               output_folder_path = result_folder_path,
                               output_file_name_root = CARA_RecommenderWrapper.RECOMMENDER_NAME)





    if flag_baselines_tune:


        ################################################################################################
        ###### Collaborative Baselines



        collaborative_algorithm_list = [
            Random,
            TopPop,
            ItemKNNCFRecommender,
            #PureSVDRecommender,
            #SLIM_BPR_Cython,
        ]


        # Running hyperparameter tuning of baslines
        # See if the results are reasonable and comparable to baselines reported in the paper
        runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                           URM_train = URM_train,
                                                           URM_train_last_test = URM_train_last_test,
                                                           metric_to_optimize = metric_to_optimize,
                                                           evaluator_validation_earlystopping = evaluator_validation,
                                                           evaluator_validation = evaluator_validation,
                                                           evaluator_test = evaluator_test,
                                                           output_folder_path = result_folder_path,
                                                           resume_from_saved = True,
                                                           parallelizeKNN = False,
                                                           allow_weighting = True,
                                                           n_cases = n_cases,
                                                           n_random_starts = n_random_starts)


        for recommender_class in collaborative_algorithm_list:
            try:
                runParameterSearch_Collaborative_partial(recommender_class)
            except Exception as e:
                print("On recommender {} Exception {}".format(recommender_class, str(e)))
                traceback.print_exc()


    ################################################################################################
    ######
    ######      PRINT RESULTS
    ######

    if flag_print_results:

        n_validation_users = 0
        n_test_users = np.sum(np.ediff1d(URM_test.indptr)>=1)

        print_time_statistics_latex_table(result_folder_path = result_folder_path,
                                          dataset_name = dataset_name,
                                          algorithm_name= ALGORITHM_NAME,
                                          other_algorithm_list = [CARA_RecommenderWrapper],
                                          KNN_similarity_to_report_list = KNN_similarity_to_report_list,
                                          n_validation_users = n_validation_users,
                                          n_test_users = n_test_users,
                                          n_decimals = 2)


        print_results_latex_table(result_folder_path = result_folder_path,
                                  algorithm_name= ALGORITHM_NAME,
                                  file_name_suffix = "article_metrics_",
                                  dataset_name = dataset_name,
                                  metrics_to_report_list = ["HIT_RATE", "NDCG"],
                                  cutoffs_to_report_list = cutoff_list_test,
                                  other_algorithm_list = [CARA_RecommenderWrapper],
                                  KNN_similarity_to_report_list = KNN_similarity_to_report_list)



        print_results_latex_table(result_folder_path = result_folder_path,
                                  algorithm_name=ALGORITHM_NAME,
                                  file_name_suffix = "all_metrics_",
                                  dataset_name = dataset_name,
                                  metrics_to_report_list = ["PRECISION", "RECALL", "MAP", "MRR", "NDCG", "F1", "HIT_RATE", "ARHR", "NOVELTY", "DIVERSITY_MEAN_INTER_LIST", "DIVERSITY_HERFINDAHL", "COVERAGE_ITEM", "DIVERSITY_GINI", "SHANNON_ENTROPY"],
                                  cutoffs_to_report_list = cutoff_list_validation,
                                  other_algorithm_list = [CARA_RecommenderWrapper],
                                  KNN_similarity_to_report_list = KNN_similarity_to_report_list)



if __name__ == '__main__':

    ALGORITHM_NAME = "CARA"
    CONFERENCE_NAME = "SIGIR"


    #parser = argparse.ArgumentParser()
    #parser.add_argument('-b', '--baseline_tune',        help="Baseline hyperparameter search", type= bool, default= False)
    #parser.add_argument('-a', '--DL_article_default',   help="Train the DL model with article hyperparameters", type= bool, default= False)
    #parser.add_argument('-p', '--print_results',        help="Print results", type= bool, default= False)


    #input_flags = parser.parse_args()
    #print(input_flags)

    input_flags = { 'baseline_tune': False, 'DL_article_default': True, 'print_results': True }

    KNN_similarity_to_report_list = ["cosine", "dice", "jaccard", "asymmetric", "tversky"]

    dataset_list = ["brightkite"]

    for dataset_name in dataset_list:
        read_data_split_and_search(dataset_name,
                                        flag_baselines_tune=input_flags['baseline_tune'],
                                        flag_DL_article_default=input_flags['DL_article_default'],
                                        flag_print_results=input_flags['print_results'],
                                        )

