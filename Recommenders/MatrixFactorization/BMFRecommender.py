#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Utils.seconds_to_biggest_unit import seconds_to_biggest_unit
import nimfa
import time


class BMFRecommender(BaseMatrixFactorizationRecommender):
    """ Binary Matrix Factorization Recommender

    https://nimfa.biolab.si/nimfa.methods.factorization.bmf.html

    """

    RECOMMENDER_NAME = "BMFRecommender"

    # https://nimfa.biolab.si/index.html#initialization-algorithms
    INIT_VALUES = ["Random", "Fixed", "NNDSVD ", "Random C", "Random VCol"]

    def __init__(self, URM_train, verbose = True):
        super(BMFRecommender, self).__init__(URM_train, verbose = verbose)


    def fit(self, num_factors = 100,
            # l1_ratio = 0.5,
            # solver = "multiplicative_update",
            init_type = "Random",
            # beta_loss = "frobenius",
            lambda_w = 1.1,
            lambda_h = 1.1,
            verbose = False,
            random_seed = None
            ):


        assert lambda_w >= 1.0, "{}: lambda_w must be >= 1.0 to ensure the resulting matrices are binary, provided value was {}".format(self.RECOMMENDER_NAME, lambda_w)
        assert lambda_h >= 1.0, "{}: lambda_h must be >= 1.0 to ensure the resulting matrices are binary, provided value was {}".format(self.RECOMMENDER_NAME, lambda_h)

        # if solver not in self.SOLVER_VALUES:
        #    raise ValueError("Value for 'solver' not recognized. Acceptable values are {}, provided was '{}'".format(self.SOLVER_VALUES.keys(), solver))
        #
        if init_type not in self.INIT_VALUES:
           raise ValueError("Value for 'init_type' not recognized. Acceptable values are {}, provided was '{}'".format(self.INIT_VALUES, init_type))
        #
        # if beta_loss not in self.BETA_LOSS_VALUES:
        #    raise ValueError("Value for 'beta_loss' not recognized. Acceptable values are {}, provided was '{}'".format(self.BETA_LOSS_VALUES, beta_loss))

        start_time = time.time()
        self._print("Computing BMF decomposition...")

        bmf_solver = nimfa.Bmf(self.URM_train,
                        seed = init_type,
                        # W = None,
                        # H = None,
                        rank = num_factors,
                        max_iter = 200, # default is 30
                        min_residuals = 1e-05,
                        # test_conv=None,
                        n_run = 1,
                        # callback=None,
                        # callback_init=None,
                        track_factor = False,
                        track_error = False,
                        lambda_w = lambda_w, #default is 1.1,
                        lambda_h = lambda_h, #default is 1.1,
                        )

        bmf_solver = bmf_solver()

        self.ITEM_factors = bmf_solver.coef().T.toarray()
        self.USER_factors = bmf_solver.basis().toarray()

        new_time_value, new_time_unit = seconds_to_biggest_unit(time.time()-start_time)
        self._print("Computing BMF decomposition... done in {:.2f} {}".format( new_time_value, new_time_unit))