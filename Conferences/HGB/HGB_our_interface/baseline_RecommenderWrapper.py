"""
Created on 02/06/22

@author: Zhitao He
"""

import dgl
import numpy as np
from types import SimpleNamespace

from Conferences.HGB.HGB_github.baseline.Model.GNN import myGAT
from Conferences.HGB.HGB_github.baseline.Model.utility.helper import ensureDir
from Conferences.HGB.HGB_github.baseline.Model.utility.loader_kgat import KGAT_loader
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.BaseTempFolder import BaseTempFolder
from Recommenders.DataIO import DataIO
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
from time import time
import torch.nn.functional as F
import torch


class baseline_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):
    RECOMMENDER_NAME = "baseline_RecommenderWrapper"

    def __init__(self, URM_train):
        # self.URM_train = URM_train
        super(baseline_RecommenderWrapper, self).__init__(URM_train)

        self.epochs = None
        self.residual = None
        self.negative_slope = None
        self.attn_drop = None
        self.feat_drop = None
        self.activation = F.elu
        self.heads = None
        self.num_layers = None
        self.num_classes = None
        self.num_hidden = None
        self.num_etypes = None
        self.edge_dim = None
        self.num_entity = None
        self.in_dim = None
        self.alpha = 0
        self.pretrain = None

        self.data_generator = None
        self.batch_size = 8196
        self.weight_decay = None
        self.g = None
        self.e_feat = None
        self.lr = None
        self.optimizer = None
        self.dataset = 'last-fm'
        self.data_path = '/home/ubuntu/Master_thesis/Conferences/HGB/HGB_github/baseline/Data/'
        self.model_type = 'baseline'
        self.layer_size = [64, 32, 16]

        args = SimpleNamespace(batch_size=self.batch_size, adj_type="si",
                               mess_dropout=[0.1], node_dropout=[0.1], layer_size=self.layer_size,
                               )

        self.data_generator = KGAT_loader(args=args,
                                          path=self.data_path + self.dataset)

        edge2type = {}
        for i, mat in enumerate(self.data_generator.lap_list):
            for u, v in zip(*mat.nonzero()):
                edge2type[(u, v)] = i
        for i in range(self.data_generator.n_users + self.data_generator.n_entities):
            edge2type[(i, i)] = len(self.data_generator.lap_list)

        adjM = sum(self.data_generator.lap_list)
        # print(len(adjM.nonzero()[0]))
        g = dgl.DGLGraph(adjM)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        e_feat = []
        edge2id = {}
        for u, v in zip(*g.edges()):
            u = u.item()
            v = v.item()
            if u == v:
                break
            e_feat.append(edge2type[(u, v)])
            edge2id[(u, v)] = len(edge2id)
        for i in range(self.data_generator.n_users + self.data_generator.n_entities):
            e_feat.append(edge2type[(i, i)])
            edge2id[(i, i)] = len(edge2id)
        e_feat = torch.tensor(e_feat, dtype=torch.long)

        self.g = g.to('cuda')
        self.e_feat = e_feat.cuda()
        self.data_generator = self.data_generator

        pre_model = 'mf'
        proj_path = '/home/ubuntu/Master_thesis/Conferences/HGB/HGB_github/baseline/Model/'
        pretrain_path = '%spretrain/%s/%s.npz' % (proj_path, self.dataset, pre_model)
        try:
            pretrain_data = np.load(pretrain_path)
            print('load the pretrained bprmf model parameters.')
        except Exception:
            pretrain_data = None

        self.pretrain = pretrain_data

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # TODO if the model in the end is either a matrix factorization algorithm or an ItemKNN/UserKNN
        #  you can have this class inherit from BaseMatrixFactorization, BaseItemSimilarityMatrixRecommender
        #  or BaseUSerSimilarityMatrixRecommender
        #  in which case you do not have to re-implement this function, you only need to set the
        #  USER_factors, ITEM_factors (see PureSVD) or W_Sparse (see ItemKNN) data structures in the FIT function
        # In order to compute the prediction the model may need a Session. The session is an attribute of this Wrapper.
        # There are two possible scenarios for the creation of the session: at the beginning of the fit function (training phase)
        # or at the end of the fit function (before loading the best model, testing phase)

        users_to_test = user_id_array

        BATCH_SIZE = self.batch_size
        ITEM_NUM = self.data_generator.n_items

        if self.model_type in ['ripple']:

            u_batch_size = BATCH_SIZE
            i_batch_size = BATCH_SIZE // 20
        elif self.model_type in ['fm', 'nfm']:
            u_batch_size = BATCH_SIZE
            i_batch_size = BATCH_SIZE
        else:
            u_batch_size = BATCH_SIZE * 2
            i_batch_size = BATCH_SIZE

        test_users = users_to_test
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1

        res = []

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_batch = test_users[start: end]

            item_batch = range(ITEM_NUM)
            with torch.no_grad():
                embedding = self.model(self.g, self.e_feat)
                user = embedding[user_batch]
                item = embedding[item_batch + self.data_generator.n_users]
                rate_batch = torch.mm(user, torch.transpose(item, 0, 1)).cpu().numpy()
                res.append(rate_batch)

        res = np.concatenate(res, axis=0)

        return res

    def _init_model(self):
        """
        This function instantiates the model, it should only rely on attributes and not function parameters
        It should be used both in the fit function and in the load_model function
        :return:
        """

        # tf.reset_default_graph()

        # TODO Instantiate the model
        # Always clear the default graph if using tensorflow

        self.model = myGAT(
            num_entity=self.num_entity,
            edge_dim=self.edge_dim,
            num_etypes=self.num_etypes,
            in_dim=self.in_dim,
            num_hidden=self.num_hidden,
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            heads=self.heads,
            activation=self.activation,
            feat_drop=self.feat_drop,
            attn_drop=self.attn_drop,
            negative_slope=self.negative_slope,
            residual=self.residual,
            pretrain=self.pretrain,
            alpha=self.alpha
        ).cuda()

    def fit(self,
            epochs=1000,

            # TODO replace those hyperparameters with the ones you need
            num_entity=62267,
            edge_dim=64,
            num_etypes=85,
            in_dim=64,
            num_hidden=32,
            num_classes=16,
            num_layers=1,
            heads=[1, 1],
            activation=F.elu,
            feat_drop=0.1,
            attn_drop=0.1,
            negative_slope=0.01,
            residual=False,
            pretrain=None,
            alpha=0.,
            batch_size=8192,
            weight_decay=1e-5,
            lr=0.0001,
            verbose=1,
            dataset=None,
            data_path=None,
            model_type='baseline',
            adj_type="si",
            mess_dropout=[0.1],
            node_dropout=[0.1],
            layer_size=[64, 32, 16],

            # These are standard
            temp_file_folder=None,
            **earlystopping_kwargs
            ):

        # Get unique temporary folder
        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)
        # self.temp_file_folder = temp_file_folder

        if num_layers is None:
            num_layers = [1, 1]

        self.epochs = epochs,
        self.num_entity = num_entity
        self.num_etypes = num_etypes
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.heads = heads
        self.activation = activation
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.residual = residual
        self.pretrain = pretrain
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.lr = lr
        self.verbose = verbose
        self.dataset = dataset
        self.data_path = data_path
        self.model_type = model_type
        self.layer_size = layer_size

        self.edge_dim = edge_dim
        self.in_dim = in_dim
        self.batch_size = batch_size

        args = SimpleNamespace(batch_size=self.batch_size, adj_type=adj_type,
                               mess_dropout=mess_dropout, node_dropout=node_dropout, layer_size=self.layer_size,
                               )

        data_generator = KGAT_loader(args=args,
                                     path=self.data_path + self.dataset)

        edge2type = {}
        for i, mat in enumerate(data_generator.lap_list):
            for u, v in zip(*mat.nonzero()):
                edge2type[(u, v)] = i
        for i in range(data_generator.n_users + data_generator.n_entities):
            edge2type[(i, i)] = len(data_generator.lap_list)

        adjM = sum(data_generator.lap_list)
        # print(len(adjM.nonzero()[0]))
        g = dgl.DGLGraph(adjM)
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        e_feat = []
        edge2id = {}
        for u, v in zip(*g.edges()):
            u = u.item()
            v = v.item()
            if u == v:
                break
            e_feat.append(edge2type[(u, v)])
            edge2id[(u, v)] = len(edge2id)
        for i in range(data_generator.n_users + data_generator.n_entities):
            e_feat.append(edge2type[(i, i)])
            edge2id[(i, i)] = len(edge2id)
        e_feat = torch.tensor(e_feat, dtype=torch.long)

        self.g = g.to('cuda')
        self.e_feat = e_feat.cuda()
        self.data_generator = data_generator

        self._init_model()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer = optimizer
        # print(self.lr)

        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.RECOMMENDER_NAME,
                                        **earlystopping_kwargs)

        self.load_model(self.temp_file_folder, file_name="_best_model")

        self._clean_temp_folder(temp_file_folder=self.temp_file_folder)

        print("{}: Training complete".format(self.RECOMMENDER_NAME))

    def _prepare_model_for_validation(self):
        # TODO Most likely you won't need to change this function
        pass

    def _update_best_model(self):
        # TODO Most likely you won't need to change this function
        self.save_model(self.temp_file_folder, file_name="_best_model")

    def _run_epoch(self, currentEpoch):
        # TODO replace this with the train loop for one epoch of the model

        t1 = time()
        loss, base_loss, kge_loss, reg_loss = 0., 0., 0., 0.
        n_batch = self.data_generator.n_train // self.batch_size + 1

        """
        *********************************************************
        Alternative Training for KGAT:
        ... phase 1: to train the recommender.
        """
        for idx in range(n_batch):
            self.model.train()
            btime = time()

            batch_data = self.data_generator.generate_train_batch()

            embedding = self.model(self.g, self.e_feat)
            u_emb = embedding[batch_data['users']]
            p_emb = embedding[batch_data['pos_items'] + self.data_generator.n_users]
            n_emb = embedding[batch_data['neg_items'] + self.data_generator.n_users]
            pos_scores = (u_emb * p_emb).sum(dim=1)
            neg_scores = (u_emb * n_emb).sum(dim=1)
            base_loss = F.softplus(-pos_scores + neg_scores).mean()
            reg_loss = self.weight_decay * ((u_emb * u_emb).sum() / 2 + (p_emb * p_emb).sum() / 2 + (
                    n_emb * n_emb).sum() / 2) / self.batch_size
            loss = base_loss + reg_loss  # Since we don't do accum, the printed loss may seem smaller
            if idx % 100 == 0:
                print(idx, loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
            currentEpoch, time() - t1, loss, base_loss, kge_loss, reg_loss)
        print(perf_str)

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        weights_save_path = '{}/{}_{}.pt'.format(folder_path + file_name,
                                                 self.num_layers, self.heads)
        ensureDir(weights_save_path)

        data_dict_to_save = {
            # TODO replace this with the hyperparameters and attribute list you need to re-instantiate
            #  the model when calling the load_model
            "num_entity": self.num_entity,
            "edge_dim": self.edge_dim,
            "num_etypes": self.num_etypes,
            "in_dim": self.in_dim,
            "num_hidden": self.num_hidden,
            "num_classes": self.num_classes,
            "num_layers": self.num_layers,
            "heads": self.heads,
            # "activation": self.activation,
            "feat_drop": self.feat_drop,
            "attn_drop": self.attn_drop,
            "negative_slope": self.negative_slope,
            "residual": self.residual,
            # "pretrain": self.pretrain,
            "alpha": self.alpha,
        }

        # Do not change this
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        torch.save(self.model, weights_save_path)

        self._print("Saving complete")

    def load_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Loading model from file '{}'".format(folder_path + file_name))

        # Reload the attributes dictionary
        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)

        for attrib_name in data_dict.keys():
            self.__setattr__(attrib_name, data_dict[attrib_name])

        weights_save_path = '{}/{}_{}.pt'.format(folder_path + file_name,
                                                 self.num_layers, self.heads)

        self._init_model()

        torch.load(weights_save_path)

        self._print("Loading complete")
