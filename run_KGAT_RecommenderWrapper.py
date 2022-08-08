import os
import traceback

import numpy as np
import tensorflow as tf
from time import time

from Conferences.HGB.HGB_github.KGAT.Model.utility.parser import parse_args
from Conferences.HGB.HGB_our_interface.DatasetProvided.MultiDatasetsReader import MultiDatasetsReader
from KGAT_RecommenderWrapper import KGAT_RecommenderWrapper
from Evaluation.Evaluator import EvaluatorHoldout


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
    tf.set_random_seed(2019)
    np.random.seed(2019)
    ALGORITHM_NAME = "KGAT"
    # CONFERENCE_NAME = "HGB"
    dataset_name = args.dataset

    result_folder_path = "result_experiments/{}/{}/".format(ALGORITHM_NAME, dataset_name)
    # model_folder_path = result_folder_path + "models/"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if dataset_name == "movie-lens" or dataset_name == "last-fm" or dataset_name == "yelp2018" or dataset_name == "amazon-book":
        dataset = MultiDatasetsReader(args.data_path + dataset_name)
    else:
        print("Dataset name not supported, current is {}".format(dataset_name))

    print('Current dataset is: {}'.format(dataset_name))
    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()

    metric_to_optimize = 'RECALL'
    cutoff_to_optimize = 20

    evaluator_validation_earlystopping = EvaluatorHoldout(URM_test, cutoff_list=[cutoff_to_optimize])

    t0 = time()

    """
    *********************************************************
    Use the pretrained data to initialize the embeddings.
    """
    if args.pretrain in [-1, -2]:
        pretrain_data = load_pretrained_data(args)
    else:
        pretrain_data = None


    try:
        # TODO fill this dictionary with the hyperparameters of the algorithm
        baseline_hyperparameters = {
            "epochs": args.epoch,
            "adj_uni_type": args.adj_uni_type,
            "lr": args.lr,
            "emb_dim": args.embed_size,
            "batch_size": args.batch_size,
            "kge_dim": args.kge_size,
            "batch_size_kg": args.batch_size_kg,
            "layer_size": args.layer_size,
            "alg_type": args.alg_type,
            "att_type": args.att_type,
            "use_ls_loss": args.use_ls_loss,
            "regs": args.regs,
            "verbose": args.verbose,
            "dataset": args.dataset,
            "data_path": args.data_path,
            "use_att": args.use_att,
            "use_kge": args.use_kge,
            "model_type": args.model_type,
            "adj_type": args.adj_type,
            "pretrain_data": pretrain_data,
            "temp_file_folder": result_folder_path,
        }

        # # Do not modify earlystopping
        earlystopping_hyperparameters = {"validation_every_n": 10,
                                         "stop_on_validation": True,
                                         "lower_validations_allowed": 10,
                                         "evaluator_object": evaluator_validation_earlystopping,
                                         "validation_metric": metric_to_optimize,
                                         }

        recommender_instance =KGAT_RecommenderWrapper(URM_train)
        recommender_instance.fit(**baseline_hyperparameters,
                                 **earlystopping_hyperparameters)

    except Exception as e:

        print("On recommender {} Exception {}".format(KGAT_RecommenderWrapper, str(e)))
        traceback.print_exc()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    read_data_split_and_search(args)
