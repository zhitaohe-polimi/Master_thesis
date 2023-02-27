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
from Utils.PyTorch.Cython.DataIterator import BPRIterator as BPRIterator_cython, \
    InteractionIterator as InteractionIterator_cython, \
    InteractionAndNegativeIterator as InteractionAndNegativeIterator_cython
from Utils.PyTorch.DataIterator import BPRIterator, InteractionIterator, InteractionAndNegativeIterator

torch.autograd.set_detect_anomaly(True)


def batch_dot(tensor_1, tensor_2):
    """
    Vectorized dot product between rows of two tensors
    :param tensor_1:
    :param tensor_2:
    :return:
    """
    # torch.einsum("ki,ki->k", tensor_1, tensor_2)
    return tensor_1.multiply(tensor_2).sum(axis=1)


class _SimpleMFModel(torch.nn.Module):

    def __init__(self, n_users, n_items, embedding_dim=20):
        super(_SimpleMFModel, self).__init__()

        self._embedding_user = torch.nn.Embedding(n_users, embedding_dim=embedding_dim)
        self._embedding_item = torch.nn.Embedding(n_items, embedding_dim=embedding_dim)
        # self._embedding_user.weight.data.uniform_(0, 0.5)
        # self._embedding_user.weight.data.uniform_(0, 0.5)

    def forward(self, user, item):
        prediction = batch_dot(self._embedding_user(user), self._embedding_item(item))
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


def loss_MSE(model, batch, l2_reg):
    user, item, rating = batch
    user = user.to("cuda")
    item = item.to("cuda")
    rating = rating.to("cuda")
    # Compute prediction for each element in batch
    prediction = model.forward(user, item)

    # Compute total loss for batch
    MSE_loss = (prediction - rating).pow(2).mean()

    reg_loss = (1 / 2) * (model._embedding_user(user).norm(2).pow(2) +
                          model._embedding_item(item).norm(2).pow(2)) / float(len(user))

    loss = MSE_loss + reg_loss * l2_reg

    return loss


def loss_CrossEntropy(model, batch):
    user, item, ground_truth = batch

    # Compute prediction for each element in batch
    predicted_class = model.forward(user, item)

    # Compute total loss for batch
    loss = torch.nn.CrossEntropyLoss()
    loss = loss(predicted_class, ground_truth)

    return loss


def loss_BPR(model, batch, l2_reg):
    user, item_positive, item_negative = batch
    user = user.to("cuda")
    item_positive = item_positive.type(torch.long).to("cuda")
    item_negative = item_negative.type(torch.long).to("cuda")

    reg_loss = (1 / 2) * (model._embedding_user(user).norm(2).pow(2) +
                          model._embedding_item(item_positive).norm(2).pow(2) +
                          model._embedding_item(item_negative).norm(2).pow(2)) / float(len(user))

    # Compute prediction for each element in batch
    x_ij = model.forward(user, item_positive) - model.forward(user, item_negative)

    # Compute total loss for batch
    BPR_loss = -x_ij.sigmoid().log().mean()

    loss = BPR_loss + reg_loss * l2_reg

    return loss


class _PyTorchMFRecommender(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):
    """
    """

    RECOMMENDER_NAME = "_PyTorchMFRecommender"

    def __init__(self, URM_train, verbose=True):
        super(_PyTorchMFRecommender, self).__init__(URM_train, verbose=verbose)

    def fit(self, epochs=300,
            batch_size=8,
            num_factors=32,
            l2_reg=1e-4,
            sgd_mode='adam',
            learning_rate=1e-2,
            **earlystopping_kwargs):

        self.l2_reg = l2_reg

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

        self._model = _SimpleMFModel(self.n_users, self.n_items, embedding_dim=num_factors)
        self._model.to("cuda")

        if sgd_mode.lower() == "adagrad":
            self._optimizer = torch.optim.Adagrad(self._model.parameters(), lr=learning_rate)  # , weight_decay=l2_reg)
        elif sgd_mode.lower() == "rmsprop":
            self._optimizer = torch.optim.RMSprop(self._model.parameters(), lr=learning_rate)  # , weight_decay=l2_reg)
        elif sgd_mode.lower() == "adam":
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)  # , weight_decay=l2_reg)
        elif sgd_mode.lower() == "sgd":
            self._optimizer = torch.optim.SGD(self._model.parameters(), lr=learning_rate)  # , weight_decay=l2_reg)
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

    def _prepare_model_for_validation(self):
        self.USER_factors = self._model._embedding_user.weight.detach().cpu().numpy()
        self.ITEM_factors = self._model._embedding_item.weight.detach().cpu().numpy()

    def _update_best_model(self):
        self.USER_factors_best = self._model._embedding_user.weight.detach().cpu().numpy()
        self.ITEM_factors_best = self._model._embedding_item.weight.detach().cpu().numpy()

    def _run_epoch(self, num_epoch):

        epoch_loss = 0

        for batch in self._data_iterator:
            # Clear previously computed gradients
            self._optimizer.zero_grad()

            # user, item_positive, item_negative = batch
            # user = user.to("cuda")
            # item_positive = item_positive.type(torch.long).to("cuda")
            # item_negative = item_negative.type(torch.long).to("cuda")
            #
            # batch = (user, item_positive, item_negative)


            # print(self._model._embedding_user(user).norm(2).pow(2),
            #       self._model._embedding_item(item_positive).norm(2).pow(2),
            #       self._model._embedding_item(item_negative).norm(2).pow(2),
            #       reg_loss,self.l2_reg)

            loss = self._loss_function(self._model, batch, self.l2_reg)

            # Compute gradients given current loss
            loss.backward()

            # Apply gradient using the selected optimizer
            self._optimizer.step()

            epoch_loss += loss.item()

        self._print("Loss {:.2E}".format(epoch_loss))


class PyTorchMF_BPR_Recommender(_PyTorchMFRecommender):
    RECOMMENDER_NAME = "PyTorchMF_BPR_Recommender"

    def __init__(self, URM_train, verbose=True):
        super(PyTorchMF_BPR_Recommender, self).__init__(URM_train, verbose=verbose)

        self._loss_function = loss_BPR


class PyTorchMF_MSE_Recommender(_PyTorchMFRecommender):
    RECOMMENDER_NAME = "PyTorchMF_MSE_Recommender"

    def __init__(self, URM_train, verbose=True):
        super(PyTorchMF_MSE_Recommender, self).__init__(URM_train, verbose=verbose)

        self.positive_quota = None
        self._loss_function = loss_MSE

    def fit(self, positive_quota=0.5, **kwargs):
        self.positive_quota = positive_quota
        super(PyTorchMF_MSE_Recommender, self).fit(**kwargs)


class PyTorchMF_Interaction_Recommender(_PyTorchMFRecommender):
    RECOMMENDER_NAME = "PyTorchMF_Interaction_Recommender"

    def __init__(self, URM_train, verbose=True):
        super(PyTorchMF_Interaction_Recommender, self).__init__(URM_train, verbose=verbose)

        self._dataset = None
        self._loss_function = loss_CrossEntropy

    def fit(self, positive_quota=0.5, **kwargs):
        self._dataset = Interaction_Dataset(self.URM_train, positive_quota=positive_quota)
        super(PyTorchMF_Interaction_Recommender, self).fit(**kwargs)
