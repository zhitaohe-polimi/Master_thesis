#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/11/2021

@author: Maurizio Ferrari Dacrema
"""

from functools import partial
import traceback
from Data_manager import *
from Recommenders.Recommender_import_list import *
from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender, BaseUserCBFRecommender
from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative, runHyperparameterSearch_Content, runHyperparameterSearch_Hybrid
import os, traceback, multiprocessing


def _get_model_list_given_dataset(dataset_class, recommender_class_list, KNN_similarity_list, ICM_dict, UCM_dict):

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

    # Removing cases that have an estimated training time of more than 30 days
    if dataset_class in [TheMoviesDatasetReader] and LightFMItemHybridRecommender in recommender_class_list:
        print("Removing 'LightFMItemHybridRecommender', estimated training time exceeds 30 days")
        model_list.remove((LightFMItemHybridRecommender, None, "ICM_all", ICM_dict["ICM_all"]))

    if dataset_class in [GowallaReader] and SLIMElasticNetRecommender in recommender_class_list:
        print("Removing 'SLIMElasticNetRecommender', estimated training time exceeds 30 days")
        model_list.remove((SLIMElasticNetRecommender, None, None, None))

    if dataset_class in [YelpReader] and LightFMCFRecommender in recommender_class_list:
        print("Removing 'LightFMCFRecommender', estimated training time exceeds 30 days")
        model_list.remove((LightFMCFRecommender, None, None, None))

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





class HyperparameterSearchEngine(object):
    """HyperparameterSearchEngine"""

    def __init__(self,
                 recommender_class_list = None,
                 KNN_similarity_list = None,
                 n_cases = None,
                 n_random_starts = None,
                 max_total_time = None,
                 metric_to_optimize = None,
                 cutoff_to_optimize = None):

        self._recommender_class_list = recommender_class_list
        self._KNN_similarity_list = KNN_similarity_list
        self._n_cases = n_cases
        self._n_random_starts = n_random_starts
        self._max_total_time = max_total_time
        self._metric_to_optimize = metric_to_optimize
        self._cutoff_to_optimize = cutoff_to_optimize


    def search(self, dataset_class = None,
               ICM_dict = None,
               UCM_dict = None,
               URM_train = None,
               URM_train_last_test = None,
               resume_from_saved = True,
               evaluator_validation = None,
               evaluator_test = None,
               evaluator_validation_earlystopping = None,
               model_folder_path = None,
               n_processes = 4):


        model_cases_list = _get_model_list_given_dataset(dataset_class,
                                                         self._recommender_class_list,
                                                         self._KNN_similarity_list,
                                                         ICM_dict,
                                                         UCM_dict)


        _optimize_single_model_partial = partial(_optimize_single_model,
                                                 URM_train = URM_train,
                                                 URM_train_last_test = URM_train_last_test,
                                                 n_cases = self._n_cases,
                                                 n_random_starts = self._n_random_starts,
                                                 resume_from_saved = resume_from_saved,
                                                 save_model = "best",
                                                 evaluate_on_test = "best",
                                                 evaluator_validation = evaluator_validation,
                                                 evaluator_test = evaluator_test,
                                                 max_total_time = self._max_total_time,
                                                 evaluator_validation_earlystopping = evaluator_validation_earlystopping,
                                                 metric_to_optimize = self._metric_to_optimize,
                                                 cutoff_to_optimize = self._cutoff_to_optimize,
                                                 model_folder_path = model_folder_path)


        pool = multiprocessing.Pool(processes=n_processes, maxtasksperchild=1)
        resultList = pool.map(_optimize_single_model_partial, model_cases_list, chunksize=1)

        pool.close()
        pool.join()

        # for single_case in model_cases_list:
        #     try:
        #         _optimize_single_model_partial(single_case)
        #     except Exception as e:
        #         print("On recommender {} Exception {}".format(single_case[0], str(e)))
        #         traceback.print_exc()





























def _get_data_split_and_folders(dataset_class):

    dataset_reader = dataset_class()

    # result_folder_path = "result_experiments/hyperopt_random_holdout_80_10_10/{}/".format(dataset_reader._get_dataset_name())
    result_folder_path = "result_experiments/hyperopt_leave_1_out/{}/".format(dataset_reader._get_dataset_name())

    data_folder_path = result_folder_path + "data/"
    model_folder_path = result_folder_path + "models/"

    # dataSplitter = DataSplitter_Holdout(dataset_reader, user_wise = False, split_interaction_quota_list=[80, 10, 10])
    dataSplitter = DataSplitter_leave_k_out(dataset_reader, k_out_value = 1, use_validation_set = True, leave_random_out = True)
    dataSplitter.load_data(save_folder_path=data_folder_path)

    # Save statistics if they do not exist
    if not os.path.isfile(data_folder_path + "item_popularity_plot"):

        URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

        plot_popularity_bias([URM_train + URM_validation, URM_test],
                             ["URM train", "URM test"],
                             data_folder_path + "item_popularity_plot")

        save_popularity_statistics([URM_train + URM_validation, URM_test],
                                   ["URM train", "URM test"],
                                   data_folder_path + "item_popularity_statistics.tex")

        all_dataset_stats_latex_table(URM_train + URM_validation + URM_test, dataset_reader._get_dataset_name(),
                                      data_folder_path + "dataset_stats.tex")

    return dataSplitter, result_folder_path, data_folder_path, model_folder_path






def _get_all_dataset_model_combinations_list(dataset_list, recommender_class_list, KNN_similarity_list):

    dataset_model_cases_list = []

    for dataset_class in dataset_list:

        dataSplitter, result_folder_path, data_folder_path, model_folder_path = _get_data_split_and_folders(dataset_class)

        model_cases_list = _get_model_list_given_dataset(dataset_class, recommender_class_list, KNN_similarity_list,
                                                         dataSplitter.get_loaded_ICM_dict(),
                                                         dataSplitter.get_loaded_UCM_dict())

        dataset_model_cases_list.extend([(dataset_class, *model_cases_tuple) for model_cases_tuple in model_cases_list])


    return dataset_model_cases_list





class HyperparameterSearchEngineParallelDataset(HyperparameterSearchEngine):
    """HyperparameterSearchEngineParallelDataset"""


    def search(self, dataset_class_list = None,
               resume_from_saved = True,
               evaluator_validation = None,
               evaluator_test = None,
               evaluator_validation_earlystopping = None,
               model_folder_path = None,
               n_processes = 4):




    dataset_model_cases_list = _get_all_dataset_model_combinations_list(dataset_list, recommender_class_list, KNN_similarity_to_report_list)















        model_cases_list = _get_model_list_given_dataset(dataset_class,
                                                         self._recommender_class_list,
                                                         self._KNN_similarity_list,
                                                         ICM_dict,
                                                         UCM_dict)


        _optimize_single_model_partial = partial(_optimize_single_model,
                                                 URM_train = URM_train,
                                                 URM_train_last_test = URM_train_last_test,
                                                 n_cases = self._n_cases,
                                                 n_random_starts = self._n_random_starts,
                                                 resume_from_saved = resume_from_saved,
                                                 save_model = "best",
                                                 evaluate_on_test = "best",
                                                 evaluator_validation = evaluator_validation,
                                                 evaluator_test = evaluator_test,
                                                 max_total_time = self._max_total_time,
                                                 evaluator_validation_earlystopping = evaluator_validation_earlystopping,
                                                 metric_to_optimize = self._metric_to_optimize,
                                                 cutoff_to_optimize = self._cutoff_to_optimize,
                                                 model_folder_path = model_folder_path)


        pool = multiprocessing.Pool(processes=n_processes, maxtasksperchild=1)
        resultList = pool.map(_optimize_single_model_partial, model_cases_list, chunksize=1)

        pool.close()
        pool.join()

        # for single_case in model_cases_list:
        #     try:
        #         _optimize_single_model_partial(single_case)
        #     except Exception as e:
        #         print("On recommender {} Exception {}".format(single_case[0], str(e)))
        #         traceback.print_exc()






