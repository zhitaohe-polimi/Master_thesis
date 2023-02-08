#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/08/2022

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
import scipy.sparse as sps
import numpy as np
import torch, os
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.preprocessing import normalize
from Utils.PyTorch.Cython.DataIterator import BPRIterator as BPRIterator_cython, InteractionIterator as InteractionIterator_cython, InteractionAndNegativeIterator as InteractionAndNegativeIterator_cython
from Utils.PyTorch.DataIterator import BPRIterator, InteractionIterator, InteractionAndNegativeIterator


def batch_dot(tensor_1, tensor_2):
    """
    Vectorized dot product between rows of two tensors
    :param tensor_1:
    :param tensor_2:
    :return:
    """
    # torch.einsum("ki,ki->k", tensor_1, tensor_2)
    return torch.einsum("ki,ki->k", tensor_1, tensor_2)  # tensor_1.multiply(tensor_2).sum(axis=1)


class _SimpleMFModel(torch.nn.Module):

    def __init__(self, n_users, n_items, embedding_dim=20):
        super(_SimpleMFModel, self).__init__()

        self._embedding_user = torch.nn.Embedding(n_users, embedding_dim=embedding_dim)
        self._embedding_item = torch.nn.Embedding(n_items, embedding_dim=embedding_dim)

    def forward(self, user, item):
        prediction = batch_dot(self._embedding_user(user), self._embedding_item(item))
        return prediction


class _SimpleNewMFModel(torch.nn.Module):

    def __init__(self, n_users, n_items, embedding_dim=20, embedding_dim_u=20, embedding_dim_i=20):
        super(_SimpleNewMFModel, self).__init__()

        self._embedding_user = torch.nn.Embedding(n_users, embedding_dim=embedding_dim)
        self._embedding_item = torch.nn.Embedding(n_items, embedding_dim=embedding_dim)

        self._embedding_user_vi = torch.nn.Embedding(n_users, embedding_dim=embedding_dim_u)
        self._embedding_item_vi = torch.nn.Embedding(n_items, embedding_dim=embedding_dim_u)

        self._embedding_user_uj = torch.nn.Embedding(n_users, embedding_dim=embedding_dim_i)
        self._embedding_item_uj = torch.nn.Embedding(n_items, embedding_dim=embedding_dim_i)

    def forward(self, user, item):
        ratings = torch.einsum("bi,ci->bc", self._embedding_user.weight, self._embedding_item.weight)

        prediction = batch_dot(self._embedding_user(user), self._embedding_item(item))

        user_sim_uv = torch.einsum("bi,ci->bc", ratings[user], ratings)
        user_sim_uv[:, user] = user_sim_uv[:, user].fill_diagonal_(0)
        user_sim_uv = torch.nn.functional.normalize(user_sim_uv, dim=1)
        alpha_vi = torch.einsum("bi,ci->bc", self._embedding_user_vi.weight, self._embedding_item_vi(item))
        summation_v = torch.einsum("bi,ib->b", user_sim_uv, alpha_vi)
        prediction += summation_v

        item_sim_ij = torch.einsum("ib,ic->bc", ratings, ratings[:, item])
        item_sim_ij[item] = item_sim_ij[item].fill_diagonal_(0)
        item_sim_ij = torch.nn.functional.normalize(item_sim_ij, dim=0)
        alpha_uj = torch.einsum("bi,ci->bc", self._embedding_user_uj(user), self._embedding_item_uj.weight)
        summation_j = torch.einsum("bi,ib->b", alpha_uj, item_sim_ij)
        prediction += summation_j

        return prediction


class _SimpleMFBiasModel(torch.nn.Module):

    def __init__(self, n_users, n_items, embedding_dim=20):
        super(_SimpleMFBiasModel, self).__init__()

        self._embedding_user = torch.nn.Embedding(n_users, embedding_dim=embedding_dim)
        self._embedding_item = torch.nn.Embedding(n_items, embedding_dim=embedding_dim)
        self._global_bias = torch.nn.Parameter(torch.randn((1), dtype=torch.float))
        self._user_bias = torch.nn.Parameter(torch.randn((n_users), dtype=torch.float))
        self._item_bias = torch.nn.Parameter(torch.randn((n_items), dtype=torch.float))

    def forward(self, user, item):
        prediction = self._global_bias + self._user_bias[user] + self._item_bias[item]
        prediction += batch_dot(self._embedding_user(user), self._embedding_item(item))
        return prediction


class _SimpleAsySVDModel(torch.nn.Module):

    def __init__(self, n_users, n_items, embedding_dim=20):
        super(_SimpleAsySVDModel, self).__init__()

        self._embedding_user_profile = torch.nn.Embedding(n_items, embedding_dim=embedding_dim)
        self._embedding_item = torch.nn.Embedding(n_items, embedding_dim=embedding_dim)

    def forward(self, user, item, user_profile):
        prediction = batch_dot(self._embedding_user(user), self._embedding_item(item))
        return prediction


class URM_Dataset(Dataset):
    def __init__(self, URM_train):
        super().__init__()
        URM_train = sps.coo_matrix(URM_train)

        self._row = torch.tensor(URM_train.row).type(torch.LongTensor)
        self._col = torch.tensor(URM_train.col).type(torch.LongTensor)
        self._data = torch.tensor(URM_train.data).type(torch.FloatTensor)

    def __len__(self):
        return len(self._row)

    def __getitem__(self, index):
        return self._row[index], self._col[index], self._data[index]


class Interaction_Dataset(Dataset):
    def __init__(self, URM_train, positive_quota):
        super().__init__()
        URM_train = sps.coo_matrix(URM_train)

        self._URM_train = sps.csr_matrix(URM_train)
        self.n_users, self.n_items = self._URM_train.shape

        self._row = torch.tensor(URM_train.row).type(torch.LongTensor)
        self._col = torch.tensor(URM_train.col).type(torch.LongTensor)
        self._data = torch.tensor(URM_train.data).type(torch.FloatTensor)
        self._positive_quota = positive_quota

    def __len__(self):
        return len(self._row)

    def __getitem__(self, index):
        select_positive_flag = torch.rand(1, requires_grad=False) > self._positive_quota

        if select_positive_flag[0]:
            return self._row[index], self._col[index], self._data[index]
        else:
            user_id = self._row[index]
            seen_items = self._URM_train.indices[self._URM_train.indptr[user_id]:self._URM_train.indptr[user_id + 1]]
            negative_selected = False

            while not negative_selected:
                negative_candidate = torch.randint(low=0, high=self.n_items, size=(1,))[0]

                if negative_candidate not in seen_items:
                    item_negative = negative_candidate
                    negative_selected = True

            return self._row[index], item_negative, torch.tensor(0.0)


class BPR_Dataset(Dataset):
    def __init__(self, URM_train):
        super().__init__()
        self._URM_train = sps.csr_matrix(URM_train)
        self.n_users, self.n_items = self._URM_train.shape

    def __len__(self):
        return self.n_users

    def __getitem__(self, user_id):

        seen_items = self._URM_train.indices[self._URM_train.indptr[user_id]:self._URM_train.indptr[user_id + 1]]
        item_positive = np.random.choice(seen_items)

        # seen_items = set(list(seen_items))
        negative_selected = False

        while not negative_selected:
            negative_candidate = np.random.randint(low=0, high=self.n_items, size=1)[0]

            if negative_candidate not in seen_items:
                item_negative = negative_candidate
                negative_selected = True

        # return torch.tensor(user_id).to("cuda"), torch.tensor(item_positive).to("cuda"), torch.tensor(item_negative).to("cuda")
        return user_id, item_positive, item_negative


def loss_MSE(model, batch):
    user, item, rating = batch
    user = user.to("cuda")
    item = item.to("cuda")
    rating = rating.to("cuda")
    # Compute prediction for each element in batch
    prediction = model.forward(user, item)

    # Compute total loss for batch
    loss = (prediction - rating).pow(2).mean()

    return loss


def loss_BPR(model, batch):
    user, item_positive, item_negative = batch
    user = user.to("cuda")
    item_positive = item_positive.type(torch.long).to("cuda")
    item_negative = item_negative.type(torch.long).to("cuda")
    # Compute prediction for each element in batch
    x_ij = model.forward(user, item_positive) - model.forward(user, item_negative)

    # Compute total loss for batch
    loss = -x_ij.sigmoid().log().mean()

    return loss


class _PyTorchMFRecommender(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):
    """
    """

    RECOMMENDER_NAME = "_PyTorchMFRecommender"

    def __init__(self, URM_train, verbose=True):
        super(_PyTorchMFRecommender, self).__init__(URM_train, verbose=verbose)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        USER_factors is n_users x n_factors
        ITEM_factors is n_items x n_factors

        The prediction for cold users will always be -inf for ALL items

        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        assert self.USER_factors.shape[1] == self.ITEM_factors.shape[1], \
            "{}: User and Item factors have inconsistent shape".format(self.RECOMMENDER_NAME)

        assert self.USER_factors.shape[0] > np.max(user_id_array), \
            "{}: Cold users not allowed. Users in trained model are {}, requested prediction for users up to {}".format(
                self.RECOMMENDER_NAME, self.USER_factors.shape[0], np.max(user_id_array))

        user_id_array = torch.Tensor(user_id_array).type(torch.LongTensor).to(self.device)
        USER_factors = torch.tensor(self.USER_factors).to(self.device)
        ITEM_factors = torch.tensor(self.ITEM_factors).to(self.device)
        USER_factors_vi = torch.tensor(self.USER_factors_vi).to(self.device)
        ITEM_factors_vi = torch.tensor(self.ITEM_factors_vi).to(self.device)
        USER_factors_uj = torch.tensor(self.USER_factors_uj).to(self.device)
        ITEM_factors_uj = torch.tensor(self.ITEM_factors_uj).to(self.device)

        if items_to_compute is not None:
            pass
            # items_to_compute_tensor = torch.Tensor(items_to_compute).type(torch.LongTensor)
            #
            # item_scores = - np.ones((len(user_id_array), ITEM_factors.shape[0]), dtype=np.float32) * np.inf
            # item_scores_t = torch.einsum("bi,ci->bc", USER_factors[user_id_array],
            #                              ITEM_factors[items_to_compute_tensor])  # .to("cuda")
            # MF_1 = torch.einsum("bi,ci->bc", USER_factors_u, ITEM_factors_u[items_to_compute_tensor])  # .to("cuda")
            # item_scores_t += torch.einsum("bi,ic->bc", users_sim[user_id_array], MF_1)  # .to("cuda")
            # MF_2 = torch.einsum("bi,ci->bc", USER_factors_i, ITEM_factors_i[items_to_compute_tensor])  # .to("cuda")
            # item_scores_t += torch.einsum("bi,ic->bc", MF_2[user_id_array],
            #                               items_sim[items_to_compute_tensor, items_to_compute_tensor])  # .to("cuda")
            # item_scores[:, items_to_compute] = item_scores_t.detach().cpu().numpy()

        else:
            item_scores = torch.einsum("bi,ci->bc", USER_factors[user_id_array], ITEM_factors)
            ratings = torch.einsum("bi,ci->bc", USER_factors, ITEM_factors)

            user_sim_uv = torch.einsum("bi,ci->bc", ratings[user_id_array], ratings)
            user_sim_uv[:, user_id_array] = user_sim_uv[:, user_id_array].fill_diagonal_(0)
            user_sim_uv = torch.nn.functional.normalize(user_sim_uv, dim=1)
            alpha_vi = torch.einsum("bi,ci->bc", USER_factors_vi, ITEM_factors_vi)
            summation_v = torch.einsum("bi,ic->bc", user_sim_uv, alpha_vi)
            item_scores += summation_v

            item_sim_ij = torch.einsum("ib,ic->bc", ratings, ratings).fill_diagonal_(0)
            item_sim_ij = torch.nn.functional.normalize(item_sim_ij, dim=0)
            alpha_uj = torch.einsum("bi,ci->bc", USER_factors_uj[user_id_array], ITEM_factors_uj)
            summation_j = torch.einsum("bi,ic->bc", alpha_uj, item_sim_ij)
            item_scores += summation_j
            item_scores = item_scores.detach().cpu().numpy()
        # No need to select only the specific negative items or warm users because the -inf score will not change
        if self.use_bias:
            item_scores += self.ITEM_bias + self.GLOBAL_bias
            item_scores = (item_scores.T + self.USER_bias[user_id_array]).T

        return item_scores

    def fit(self, epochs=300,
            batch_size=8,
            num_factors=32,
            num_factors_u=32,
            num_factors_i=32,
            l2_reg=1e-4,
            sgd_mode='adam',
            learning_rate=1e-2,
            **earlystopping_kwargs):

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("MF_MSE_PyTorch: Using CUDA")
        else:
            self.device = torch.device('cpu')
            print("MF_MSE_PyTorch: Using CPU")

        use_cython_sampler = True

        if self.RECOMMENDER_NAME == "PyTorchMF_BPR_Recommender":
            data_iterator_class = BPRIterator_cython if use_cython_sampler else BPRIterator
            self._data_iterator = data_iterator_class(URM_train=self.URM_train, batch_size=batch_size)
        elif self.RECOMMENDER_NAME == "PyTorchMF_MSE_Recommender":
            data_iterator_class = InteractionIterator_cython if use_cython_sampler else InteractionIterator
            self._data_iterator = data_iterator_class(URM_train=self.URM_train, positive_quota=self.positive_quota,
                                                      batch_size=batch_size)
        else:
            self._data_iterator = None

        # self._data_loader = DataLoader(self._dataset, batch_size=int(batch_size), shuffle=True,
        #                                num_workers=os.cpu_count(), pin_memory=True)

        self._model = _SimpleNewMFModel(self.n_users, self.n_items, embedding_dim=num_factors,
                                        embedding_dim_u=num_factors_u, embedding_dim_i=num_factors_i)

        self._model = self._model.to(self.device)

        print("ITERACTIONS OF URM_TRAIN(fit): ", self.URM_train.nnz)

        # self.users_sim = torch.nn.functional.normalize(self.users_sim, dim=1)
        # print("user similarity computed.")

        if sgd_mode.lower() == "adagrad":
            self._optimizer = torch.optim.Adagrad(self._model.parameters(), lr=learning_rate, weight_decay=l2_reg)
        elif sgd_mode.lower() == "rmsprop":
            self._optimizer = torch.optim.RMSprop(self._model.parameters(), lr=learning_rate, weight_decay=l2_reg)
        elif sgd_mode.lower() == "adam":
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate, weight_decay=l2_reg)
        elif sgd_mode.lower() == "sgd":
            self._optimizer = torch.optim.SGD(self._model.parameters(), lr=learning_rate, weight_decay=l2_reg)
        else:
            raise ValueError("sgd_mode attribute value not recognized.")

        self._update_best_model()

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
        # prof.export_chrome_trace("trace.json")

        self._print("Training complete")

        self.USER_factors = self.USER_factors_best.copy()
        self.ITEM_factors = self.ITEM_factors_best.copy()

        self.USER_factors_vi = self.USER_factors_best_vi.copy()
        self.ITEM_factors_vi = self.ITEM_factors_best_vi.copy()

        self.USER_factors_uj = self.USER_factors_best_uj.copy()
        self.ITEM_factors_uj = self.ITEM_factors_best_uj.copy()

    def _prepare_model_for_validation(self):
        self.USER_factors = self._model._embedding_user.weight.detach().cpu().numpy()
        self.ITEM_factors = self._model._embedding_item.weight.detach().cpu().numpy()

        self.USER_factors_vi = self._model._embedding_user_vi.weight.detach().cpu().numpy()
        self.ITEM_factors_vi = self._model._embedding_item_vi.weight.detach().cpu().numpy()

        self.USER_factors_uj = self._model._embedding_user_uj.weight.detach().cpu().numpy()
        self.ITEM_factors_uj = self._model._embedding_item_uj.weight.detach().cpu().numpy()

    def _update_best_model(self):
        self.USER_factors_best = self._model._embedding_user.weight.detach().cpu().numpy()
        self.ITEM_factors_best = self._model._embedding_item.weight.detach().cpu().numpy()

        self.USER_factors_best_vi = self._model._embedding_user_vi.weight.detach().cpu().numpy()
        self.ITEM_factors_best_vi = self._model._embedding_item_vi.weight.detach().cpu().numpy()

        self.USER_factors_best_uj = self._model._embedding_user_uj.weight.detach().cpu().numpy()
        self.ITEM_factors_best_uj = self._model._embedding_item_uj.weight.detach().cpu().numpy()

    def _run_epoch(self, num_epoch):

        epoch_loss = 0
        for batch in self._data_iterator:
            # Clear previously computed gradients
            self._optimizer.zero_grad()

            loss = self._loss_function(self._model, batch)

            # Compute gradients given current loss
            loss.backward()

            # Apply gradient using the selected optimizer
            self._optimizer.step()

            epoch_loss += loss.item()

        self._print("Loss {:.2E}".format(epoch_loss))


class PyTorchNewMF_BPR_Recommender(_PyTorchMFRecommender):
    RECOMMENDER_NAME = "PyTorchNewMF_BPR_Recommender"

    def __init__(self, URM_train, verbose=True):
        super(PyTorchNewMF_BPR_Recommender, self).__init__(URM_train, verbose=verbose)

        # self._dataset = BPR_Dataset(self.URM_train)
        self._loss_function = loss_BPR


class PyTorchNewMF_MSE_Recommender(_PyTorchMFRecommender):
    RECOMMENDER_NAME = "PyTorchNewMF_MSE_Recommender"

    def __init__(self, URM_train, verbose=True):
        super(PyTorchNewMF_MSE_Recommender, self).__init__(URM_train, verbose=verbose)

        self.positive_quota = None
        self._loss_function = loss_MSE

    def fit(self, positive_quota=0.5, **kwargs):
        self.positive_quota = positive_quota
        super(PyTorchNewMF_MSE_Recommender, self).fit(**kwargs)
