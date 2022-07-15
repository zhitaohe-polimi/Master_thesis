#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""


from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Recommender_utils import check_matrix

from Recommenders.Similarity.Compute_Similarity import Compute_Similarity


import subprocess
import os, sys, time

import numpy as np



def default_validation_function(self):


    return self.evaluateRecommendations(self.URM_validation)


class SLIM_Structure_Cython(BaseItemSimilarityMatrixRecommender):

    RECOMMENDER_NAME = "SLIM_Structure_Recommender"

    INIT_TYPE_VALUES = ["random", "one", "zero", "copy_similarity"]
    STRUCTURE_MODE_VALUES = ["full", "similarity"]#, "common_feature"]


    def __init__(self, URM_train, ICM = None, URM_validation = None):


        super(SLIM_Structure_Cython, self).__init__()


        self.URM_train = URM_train.copy()
        self.n_users = URM_train.shape[0]
        self.n_items = URM_train.shape[1]
        self.normalize = False
        self.sparse_weights = True

        if URM_validation is not None:
            self.URM_validation = URM_validation.copy()
        else:
            self.URM_validation = None

        if ICM is not None:
            self.ICM = ICM.copy()
        else:
            self.ICM = None


        self.URM_mask = self.URM_train.copy()




    def fit(self, epochs=300, logFile=None, URM_test=None, filterTopPop = False, minRatingsPerUser=1,
            batch_size = 1, lambda_1 = 0.0, lambda_2 = 0.0, learning_rate = 1e-3, topK = 200,
            structure_mode = "full", init_type = "random", loss = "mse", force_positive = False,
            sgd_mode='adam', gamma=0.995, beta_1=0.9, beta_2=0.999,
            stop_on_validation = False, lower_validations_allowed = 5, validation_metric = "map",
            validation_function = None, validation_every_n = 1):



        # Import compiled module
        from SLIM_ElasticNet.Cython.SLIM_Structure_Cython_Epoch import SLIM_Structure_Cython_Epoch

        if(topK != False and topK<1):
            raise ValueError("TopK not valid. Acceptable values are either False or a positive integer value. Provided value was '{}'".format(topK))
        self.topK = topK

        if self.topK == False:
            self.sparse_weights = False
        else:
            self.sparse_weights = True


        if structure_mode not in self.STRUCTURE_MODE_VALUES:
            raise ValueError("Value for 'structure_mode' not recognized. Acceptable values are {}, provided was '{}'".format(self.STRUCTURE_MODE_VALUES, structure_mode))

        if init_type not in self.INIT_TYPE_VALUES:
           raise ValueError("Value for 'init_type' not recognized. Acceptable values are {}, provided was '{}'".format(self.INIT_TYPE_VALUES, init_type))

        if init_type == "zero" and loss == "mse":
            print("WARNING: If loss is 'mse' init_type cannot be 'zero' as SGD would not be able to move."
                  "Forcing init_type to 'random'")

            init_type = "random"


        self.sgd_mode = sgd_mode
        self.epochs = epochs

        if structure_mode == "similarity":
            similarity = Compute_Similarity(self.ICM.T, shrink=0, topK=topK, normalize=False, mode = "cosine")
            S_structure = similarity.compute_similarity()

            #self.structural_statistics(self.URM_train, S_structure)

        else:
            S_structure = None

        self.cythonEpoch = SLIM_Structure_Cython_Epoch(URM = self.URM_train,
                                                       S_structure = S_structure,
                                                       ICM = self.ICM,
                                                       learning_rate=learning_rate,
                                                       topK = self.topK,
                                                       batch_size = batch_size,
                                                       lambda_1 = lambda_1,
                                                       lambda_2 = lambda_2,
                                                       init_type = init_type,
                                                       structure_mode = structure_mode,
                                                       sgd_mode=sgd_mode,
                                                       gamma=gamma,
                                                       beta_1=beta_1,
                                                       beta_2=beta_2)





        self.logFile = logFile

        cython_epochs = self.epochs

        if validation_every_n is not None:
            self.validation_every_n = validation_every_n
            cython_epochs = validation_every_n
        else:
            self.validation_every_n = np.inf

        if validation_function is None:
            validation_function = default_validation_function


        self.learning_rate = learning_rate


        start_time = time.time()


        best_validation_metric = None
        lower_validatons_count = 0
        convergence = False

        self.S_incremental = self.cythonEpoch.get_S()
        self.S_best = self.S_incremental.copy()
        self.epochs_best = 0

        currentEpoch = 0



        # self.get_S_incremental_and_set_W()
        # results_run = validation_function(self)
        # print("SLIM_Structure_Cython: {}".format(results_run))



        while currentEpoch < self.epochs and not convergence:

            self.cythonEpoch.epochIteration_Cython(epochs = cython_epochs, loss = loss, force_positive = force_positive)
            currentEpoch += cython_epochs

            # Determine whether a validaton step is required
            if self.URM_validation is not None and (currentEpoch) % self.validation_every_n == 0:

                print("SLIM_Structure_Cython: Validation begins...")

                self.get_S_incremental_and_set_W()

                results_run = validation_function(self)

                print("SLIM_Structure_Cython: {}".format(results_run))

                # Update the D_best and V_best
                # If validation is required, check whether result is better
                if stop_on_validation:

                    current_metric_value = results_run[validation_metric]

                    if best_validation_metric is None or best_validation_metric < current_metric_value:

                        best_validation_metric = current_metric_value
                        self.S_best = self.S_incremental.copy()
                        self.epochs_best = currentEpoch

                    else:
                        lower_validatons_count += 1

                    if lower_validatons_count >= lower_validations_allowed:
                        convergence = True
                        print("SLIM_Structure_Cython: Convergence reached! Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} min".format(
                            currentEpoch, validation_metric, self.epochs_best, best_validation_metric, (time.time() - start_time) / 60))


            # If no validation required, always keep the latest
            if not stop_on_validation:
                self.S_best = self.S_incremental.copy()

            print("SLIM_Structure_Cython: Epoch {} of {}. Elapsed time {:.2f} min".format(
                currentEpoch, self.epochs, (time.time() - start_time) / 60))


        self.get_S_incremental_and_set_W()

        sys.stdout.flush()


    def structural_statistics(self, URM_train, S_structure):

        print("SLIM_Structure_Cython: computing structural_statistics...")

        import matplotlib.pyplot as plt

        # Average quota of user interaction associated to similarities within structure
        global_interactions = 0
        usable_interactions = 0

        S_structure = S_structure.copy()
        S_structure = check_matrix(S_structure, "csc")
        S_structure.data = np.ones_like(S_structure.data)

        URM_train = URM_train.copy()
        URM_train.data = np.ones_like(URM_train.data)

        usable_interactions_per_item = np.zeros(self.n_items)
        interactions_per_item =  np.zeros(self.n_items)

        URM_train_csc = check_matrix(URM_train, "csc")
        URM_train_csr = check_matrix(URM_train, "csr")

        for current_item in range(self.n_items):

            item_samples = URM_train_csc[:,current_item]

            user_interactions_per_item = URM_train_csr[item_samples.indices,:].sum(axis=0)

            usable_interactions_per_item[current_item] = np.multiply(S_structure[:,current_item].toarray().ravel(), np.array(user_interactions_per_item).ravel()).sum()

            interactions_per_item[current_item] = user_interactions_per_item.sum()


        usable_interactions = usable_interactions_per_item.sum()
        global_interactions = interactions_per_item.sum()

        print("Global interactions {:.2E}, usable {:.2E} ( {:.2f} %)".format(
            global_interactions, usable_interactions, usable_interactions/global_interactions*100))

        # Turn interactive plotting off
        plt.ioff()

        # Ensure it works even on SSH
        plt.switch_backend('agg')

        # plt.xlabel('Item')
        # plt.ylabel("% of usable interactions")
        # plt.title("Proportion of usable interactions per item in %")
        #
        # interactions_per_item[interactions_per_item==0] = np.nan
        #
        # proportion_of_usable_interactions = - np.sort(-usable_interactions_per_item/interactions_per_item*100)
        #
        # plt.plot(np.arange(self.n_items), proportion_of_usable_interactions, linewidth=3, label="CBF data",
        #          linestyle = ":")
        #
        # plt.legend()
        #



        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Item')
        ax1.set_ylabel("Interactions")
        plt.title("Item interactions and usable interactions")

        warm_item_mask = interactions_per_item!=0
        n_warm_items = warm_item_mask.sum()
        interactions_per_item = interactions_per_item[warm_item_mask]
        usable_interactions_per_item = usable_interactions_per_item[warm_item_mask]


        proportion_of_usable_interactions = usable_interactions_per_item/interactions_per_item*100

        sorted_index = np.argsort(-proportion_of_usable_interactions)

        proportion_of_usable_interactions = proportion_of_usable_interactions[sorted_index]
        interactions_per_item = interactions_per_item[sorted_index]


        plot1, = ax1.plot(np.arange(n_warm_items), proportion_of_usable_interactions, linewidth=3, label="Usable interactions",
                 linestyle = "-", c="r")

        ax2 = ax1.twinx()

        plot2, = ax2.plot(np.arange(n_warm_items), interactions_per_item, linewidth=3, label="Global interactions",
                 linestyle = ":")


        plt.legend((plot1, plot2,), ("Usable interactions", "Global interactions"))


        plt.savefig("./result_experiments/Proportion_of_usable_interactions_per_item")

        plt.close()

        print("SLIM_Structure_Cython: computing structural_statistics... complete")




    def get_S_incremental_and_set_W(self):

        self.S_incremental = self.cythonEpoch.get_S()

        if self.sparse_weights:
            self.W_sparse = self.S_incremental.copy()
        else:
            self.W = self.S_incremental






class SLIM_Structure_BPR_Cython(SLIM_Structure_Cython):
    """
    Subclas allowing only for SLIM BPR
    """

    RECOMMENDER_NAME = "SLIM_Structure_BPR_Recommender"

    def __init__(self, *pos_args, **key_args):

        super(SLIM_Structure_BPR_Cython, self).__init__(*pos_args, **key_args)

    def fit(self, **key_args):

        if "loss" in key_args and key_args["loss"] != "bpr":
            print("SLIM_Structure_BPR_Cython: 'loss' value not acceptable, forcing to 'bpr'")

        key_args["loss"] = "bpr"
        super(SLIM_Structure_BPR_Cython, self).fit(**key_args)




class SLIM_Structure_MSE_Cython(SLIM_Structure_Cython):
    """
    Subclas allowing only for SLIM MSE
    """

    RECOMMENDER_NAME = "SLIM_Structure_MSE_Recommender"

    def __init__(self, *pos_args, **key_args):

        super(SLIM_Structure_MSE_Cython, self).__init__(*pos_args, **key_args)

    def fit(self, **key_args):

        if "loss" in key_args and key_args["loss"] != "mse":
            print("SLIM_Structure_MSE_Cython: 'loss' value not acceptable, forcing to 'mse'")

        key_args["loss"] = "mse"
        super(SLIM_Structure_MSE_Cython, self).fit(**key_args)
