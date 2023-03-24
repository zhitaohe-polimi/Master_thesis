import argparse
import multiprocessing
import os
import sys
import traceback
from functools import partial

import numpy as np
import torch

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append('/home')

from Conferences.HGB.HGB_github.baseline.Model.utility.helper import *
from Conferences.HGB.HGB_github.baseline.Model.utility.loader_kgat import KGAT_loader
from time import time

from Conferences.HGB.HGB_our_interface.DatasetProvided.MultiDatasetsReader import MultiDatasetsReader
from Conferences.HGB.HGB_our_interface.baseline_RecommenderWrapper import baseline_RecommenderWrapper
from Data_manager.Movielens.Movielens1MReader import Movielens1MReader
from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout

from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
# from Recommenders.MatrixFactorization.Cython.new_algo_with_MFAttention_Cython import \
#     new_MatrixFactorization_FunkSVD_Cython
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython, \
    MatrixFactorization_BPR_Cython,MatrixFactorization_AsySVD_Cython
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.MatrixFactorization.PyTorchNewMF import PyTorchNewMF_MSE_Recommender, PyTorchNewMF_BPR_Recommender
from Recommenders.MatrixFactorization.PyTorchMF import PyTorchMF_MSE_Recommender,PyTorchMF_BPR_Recommender
from Conferences.HGB.HGB_our_interface.customized_PureSVDRecommender import customized_PureSVDRecommender
# from Utils.ResultFolderLoader import ResultFolderLoader
from Utils.assertions_on_data_for_experiments import assert_implicit_data, assert_disjoint_matrices
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython


def load_pretrained_data(args):
    pre_model = 'mf'
    if args.pretrain == -2:
        pre_model = 'kgat'
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, pre_model)
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained bprmf model parameters.')
    except Exception:
        pretrain_data = None
    return pretrain_data


def read_data_split_and_search(args):
    torch.manual_seed(2021)
    np.random.seed(2019)

    ALGORITHM_NAME = "baseline"
    # CONFERENCE_NAME = "HGB"
    dataset_name = args.dataset

    metric_to_optimize = 'RECALL'  # 'NDCG' 'RECALL'
    result_folder_path = "result_experiments/{}/{}/".format(ALGORITHM_NAME, dataset_name)
    model_folder_path = result_folder_path + "models_%s/" % metric_to_optimize

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if dataset_name == "movie-lens" or dataset_name == "last-fm" or dataset_name == "yelp2018" or dataset_name == "amazon-book":
        dataset = MultiDatasetsReader(args.data_path + dataset_name)
        # dataReader = Movielens1MReader()
        # dataset = dataReader.load_data()

    else:
        print("Dataset name not supported, current is {}".format(dataset_name))

    print('Current dataset is: {}'.format(dataset_name))
    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()

    # URM_train, URM_test = split_train_in_two_percentage_global_sample(dataset.get_URM_all(), train_percentage=0.80)
    # URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

    URM_train_original = URM_train + URM_validation

    # Ensure IMPLICIT data and disjoint test-train split
    # assert_disjoint_matrices([URM_train, URM_validation, URM_test])

    cutoff_to_optimize = 20
    cutoff_list = [20]

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list)
    evaluator_validation_earlystopping = EvaluatorHoldout(URM_validation, cutoff_list=[cutoff_to_optimize])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    # This is for paper baseline algorithms
    evaluator_test_earlystopping = EvaluatorHoldout(URM_test, cutoff_list=[cutoff_to_optimize])

    if args.flag_algo_article_default:
        data_generator = KGAT_loader(args=args, path=args.data_path + args.dataset)

        """
        *********************************************************
        Load Data from data_generator function.
        """
        config = dict()
        config['n_users'] = data_generator.n_users
        config['n_items'] = data_generator.n_items
        config['n_relations'] = data_generator.n_relations
        config['n_entities'] = data_generator.n_entities

        if args.model_type in ['kgat', 'cfkg']:
            "Load the laplacian matrix."
            config['A_in'] = sum(data_generator.lap_list)

            "Load the KG triplets."
            config['all_h_list'] = data_generator.all_h_list
            config['all_r_list'] = data_generator.all_r_list
            config['all_t_list'] = data_generator.all_t_list
            config['all_v_list'] = data_generator.all_v_list

        t0 = time()

        """
        *********************************************************
        Use the pretrained data to initialize the embeddings.
        """
        if args.pretrain in [-1, -2]:
            pretrain_data = load_pretrained_data(args)
        else:
            pretrain_data = None

        weight_size = eval(args.layer_size)
        num_layers = len(weight_size) - 2
        heads = [args.heads] * num_layers + [1]

        # print(config['n_users'] + config['n_entities'],args.kge_size,config['n_relations'] * 2 + 1,
        #       args.embed_size,weight_size[-2],weight_size[-1],num_layers,heads,
        #       args.batch_size,args.verbose,args.lr,args.layer_size)

        try:
            # TODO fill this dictionary with the hyperparameters of the algorithm
            baseline_hyperparameters = {
                "epochs": args.epoch,
                "num_entity": config['n_users'] + config['n_entities'],
                "edge_dim": args.kge_size,
                "num_etypes": config['n_relations'] * 2 + 1,
                "in_dim": args.embed_size,
                "num_hidden": weight_size[-2],
                "num_classes": weight_size[-1],
                "num_layers": num_layers,
                "heads": heads,
                "pretrain": pretrain_data,
                "batch_size": args.batch_size,
                "verbose": args.verbose,
                "lr": args.lr,
                "dataset": args.dataset,
                "data_path": args.data_path,
                "model_type": args.model_type,
                "layer_size": args.layer_size,
                "temp_file_folder": result_folder_path,
            }

            # # Do not modify earlystopping
            earlystopping_hyperparameters = {"validation_every_n": 10,
                                             "stop_on_validation": True,
                                             "lower_validations_allowed": 10,
                                             "evaluator_object": evaluator_test_earlystopping,
                                             "validation_metric": metric_to_optimize,
                                             }

            recommender_instance = baseline_RecommenderWrapper(URM_train_original)
            recommender_instance.fit(**baseline_hyperparameters, **earlystopping_hyperparameters)

        except Exception as e:

            print("On recommender {} Exception {}".format(baseline_RecommenderWrapper, str(e)))
            traceback.print_exc()
    ################################################################################################
    ######
    ######      BASELINE ALGORITHMS - Nothing should be modified below this point
    ######

    if args.flag_baselines_tune:
        recommender_class_list = [
            # P3alphaRecommender,
            # SLIM_BPR_Cython,
            # MatrixFactorization_BPR_Cython,
            # IALSRecommender,
            # MatrixFactorization_FunkSVD_Cython,
            # MatrixFactorization_AsySVD_Cython,
            # EASE_R_Recommender,
            # ItemKNNCFRecommender,
            # UserKNNCFRecommender,
            # PureSVDRecommender,
            # customized_PureSVDRecommender,
            # UserKNNCBFRecommender,
            # ItemKNNCBFRecommender,
            # new_MatrixFactorization_FunkSVD_Cython,
            # PyTorchNewMF_MSE_Recommender,
            PyTorchNewMF_BPR_Recommender,
            # PyTorchMF_BPR_Recommender,
            # PyTorchMF_MSE_Recommender
        ]

        n_cases = 300

        # runParameterSearch_Collaborative_partial = runHyperparameterSearch_Collaborative(PyTorchNewMF_BPR_Recommender,
        #                                                    URM_train=URM_train,
        #                                                    URM_train_last_test=URM_train_original,
        #                                                    metric_to_optimize=metric_to_optimize,
        #                                                    cutoff_to_optimize=cutoff_to_optimize,
        #                                                    n_cases=n_cases,
        #                                                    n_random_starts=int(n_cases / 3),
        #                                                    evaluator_validation_earlystopping=evaluator_validation,
        #                                                    evaluator_validation=evaluator_validation,
        #                                                    evaluate_on_test='best',
        #                                                    evaluator_test=evaluator_test,
        #                                                    output_folder_path=model_folder_path,
        #                                                    resume_from_saved=True,
        #                                                    similarity_type_list=None,  # all
        #                                                    parallelizeKNN=False)
        #
        runParameterSearch_Collaborative_partial = partial(runHyperparameterSearch_Collaborative,
                                                           URM_train=URM_train,
                                                           URM_train_last_test=URM_train_original,
                                                           metric_to_optimize=metric_to_optimize,
                                                           cutoff_to_optimize=cutoff_to_optimize,
                                                           n_cases=n_cases,
                                                           n_random_starts=int(n_cases / 3),
                                                           evaluator_validation_earlystopping=evaluator_validation,
                                                           evaluator_validation=evaluator_validation,
                                                           evaluate_on_test='best',
                                                           evaluator_test=evaluator_test,
                                                           output_folder_path=model_folder_path,
                                                           resume_from_saved=True,
                                                           similarity_type_list=None,  # all
                                                           parallelizeKNN=False)
        multiprocessing.set_start_method('spawn')
        pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()-4), maxtasksperchild=1)
        pool.map(runParameterSearch_Collaborative_partial, recommender_class_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KGAT.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='/home/tesista/Master_thesis'
                                                          '/Conferences/HGB/HGB_github/baseline/Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='/home/tesista/Master_thesis'
                                                          '/Conferences/HGB/HGB_github/baseline/Model/',
                        help='Project path.')

    parser.add_argument('--dataset', help="Choose a dataset from {yelp2018, last-fm, amazon-book}",
                        default="amazon-book")

    parser.add_argument('--pretrain', type=int, default=-1,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')

    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')

    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='CF Embedding size.')

    parser.add_argument('--kge_size', type=int, default=64,
                        help='KG Embedding size.')

    parser.add_argument('--layer_size', nargs='?', default='[64,32,16]',
                        help='Output sizes of every layer')

    parser.add_argument('--batch_size', type=int, default=8192,
                        help='CF batch size.')

    parser.add_argument('--batch_size_kg', type=int, default=2048,
                        help='KG batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularization for user and item embeddings.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='baseline',
                        help='Specify a loss type from {kgat, bprmf, fm, nfm, cke, cfkg}.')
    parser.add_argument('--adj_type', nargs='?', default='si',
                        help='Specify the type of the adjacency (laplacian) matrix from {bi, si}.')
    parser.add_argument('--alg_type', nargs='?', default='ngcf',
                        help='Specify the type of the graph convolutional layer from {bi, gcn, graphsage}.')
    parser.add_argument('--adj_uni_type', nargs='?', default='sum',
                        help='Specify a loss type (uni, sum).')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    parser.add_argument('--use_att', type=bool, default=False,
                        help='whether using attention mechanism')
    parser.add_argument('--use_kge', type=bool, default=False,
                        help='whether using knowledge graph embedding')

    parser.add_argument('--l1_flag', type=bool, default=False,
                        help='Flase: using the L2 norm, True: using the L1 norm.')
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.)

    parser.add_argument('--flag_algo_article_default', type=bool, default=False)
    parser.add_argument('--flag_baselines_tune', type=bool, default=True)
    parser.add_argument('--flag_print_results', type=bool, default=True)

    input_flags = parser.parse_args()
    print(input_flags)

    read_data_split_and_search(input_flags)
