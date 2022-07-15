"""
Created on 21/06/22

@author: Zhitao He
"""

# TODO replace the recommender class name with the correct one
import os
import sys
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from time import time

from Conferences.HGB.HGB_github.KGAT.Model.KGAT import KGAT
# from Conferences.HGB.HGB_github.KGAT.Model.utility.batch_test import test
from Conferences.HGB.HGB_github.KGAT.Model.utility.helper import ensureDir
from Conferences.HGB.HGB_github.KGAT.Model.utility.loader_kgat import KGAT_loader
from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.BaseTempFolder import BaseTempFolder
from Recommenders.DataIO import DataIO
from Recommenders.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping


class KGAT_RecommenderWrapper(BaseRecommender, Incremental_Training_Early_Stopping, BaseTempFolder):
    # TODO replace the recommender name with the correct one
    RECOMMENDER_NAME = "KGAT_RecommenderWrapper"

    def __init__(self, URM_train):
        # TODO remove ICM_train and inheritance from BaseItemCBFRecommender if content features are not needed
        super(KGAT_RecommenderWrapper, self).__init__(URM_train)
        self.mess_dropout = None
        self.model_type = None
        self.data_path = None
        self.dataset = None
        self.use_kge = None
        self.use_att = None
        self.regs = None
        self.use_ls_loss = None
        self.att_type = None
        self.alg_type = None
        self.layer_size = None
        self.batch_size_kg = None
        self.kge_dim = None
        self.batch_size = None
        self.emb_dim = None
        self.lr = None
        self.adj_uni_type = None
        self.sess = tf.Session()

        self.batch_test_flag = None
        self.data_generator = None
        self.args = None
        self.pretrain_data = None
        self.config = None

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        users_to_test = user_id_array
        drop_flag = False

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

        count = 0

        res = []

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_batch = test_users[start: end]

            if self.batch_test_flag:

                n_item_batchs = ITEM_NUM // i_batch_size + 1
                rate_batch = np.zeros(shape=(len(user_batch), ITEM_NUM))

                i_count = 0
                for i_batch_id in range(n_item_batchs):
                    i_start = i_batch_id * i_batch_size
                    i_end = min((i_batch_id + 1) * i_batch_size, ITEM_NUM)

                    item_batch = range(i_start, i_end)

                    feed_dict = self.data_generator.generate_test_feed_dict(model=self.model,
                                                                            user_batch=user_batch,
                                                                            item_batch=item_batch,
                                                                            drop_flag=drop_flag)
                    i_rate_batch = self.model.eval(self.sess, feed_dict=feed_dict)
                    i_rate_batch = i_rate_batch.reshape((-1, len(item_batch)))

                    rate_batch[:, i_start: i_end] = i_rate_batch
                    i_count += i_rate_batch.shape[1]

                assert i_count == ITEM_NUM

            else:
                item_batch = range(ITEM_NUM)
                feed_dict = self.data_generator.generate_test_feed_dict(model=self.model,
                                                                        user_batch=user_batch,
                                                                        item_batch=item_batch,
                                                                        drop_flag=drop_flag)
                rate_batch = self.model.eval(self.sess, feed_dict=feed_dict)
                rate_batch = rate_batch.reshape((-1, len(item_batch)))

            res.append(rate_batch)

        res = np.concatenate(res, axis=0)

        return res

    def _init_model(self):
        """
        This function instantiates the model, it should only rely on attributes and not function parameters
        It should be used both in the fit function and in the load_model function
        :return:
        """

        tf.reset_default_graph()

        # TODO Instantiate the model
        # Always clear the default graph if using tensorflow
        self.model = KGAT(data_config=self.config, pretrain_data=self.pretrain_data, args=self.args)

    def fit(self,
            epochs=1000,
            # TODO replace those hyperparameters with the ones you need
            pretrain_data=None,
            adj_uni_type="sum",
            lr=0.0001,
            emb_dim=64,
            batch_size=1024,
            kge_dim=64,
            batch_size_kg=2048,
            layer_size="[64,32,16]",
            alg_type="bi",
            att_type="kgat",
            use_ls_loss=False,
            regs="[1e-5,1e-5]",
            verbose=50,
            use_att=True,
            use_kge=True,
            model_type="kgat",
            dataset=None,
            data_path=None,
            adj_type="si",
            node_dropout="[0.1]",
            mess_dropout="[0.1, 0.1, 0.1]",
            no_rel_type=False,

            # These are standard
            temp_file_folder=None,
            **earlystopping_kwargs
            ):

        # Get unique temporary folder
        self.temp_file_folder = self._get_unique_temp_folder(input_temp_file_folder=temp_file_folder)

        self.pretrain_data = pretrain_data
        self.adj_uni_type = adj_uni_type
        self.lr = lr
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.kge_dim = kge_dim
        self.batch_size_kg = batch_size_kg
        self.layer_size = layer_size
        self.alg_type = alg_type
        self.att_type = att_type
        self.use_ls_loss = use_ls_loss
        self.regs = regs
        self.verbose = verbose
        self.use_att = use_att
        self.use_kge = use_kge
        self.dataset = dataset
        self.data_path = data_path
        self.model_type = model_type

        print(self.pretrain_data,adj_type,self.adj_uni_type,self.lr,self.emb_dim,self.batch_size,self.kge_dim,
              self.batch_size_kg,self.layer_size,self.alg_type,self.att_type,self.use_ls_loss,self.regs,
              self.verbose,self.use_att,self.use_kge,self.dataset,self.data_path,self.model_type)

        args = SimpleNamespace(batch_size=self.batch_size, adj_type=adj_type,
                               mess_dropout=mess_dropout, node_dropout=node_dropout, layer_size=self.layer_size,
                               adj_uni_type=self.adj_uni_type, lr=self.lr, embed_size=self.emb_dim,
                               kge_size=self.kge_dim, batch_size_kg=self.batch_size_kg,
                               alg_type=self.alg_type, att_type=self.att_type, use_ls_loss=self.use_ls_loss,
                               regs=self.regs, verbose=self.verbose, no_rel_type=no_rel_type
                               )
        self.args = args

        # saver = tf.train.Saver()
        data_generator = KGAT_loader(args=self.args, path=self.data_path + self.dataset)
        self.batch_test_flag = False

        self.data_generator = data_generator

        """
        *********************************************************
        Load Data from data_generator function.
        """

        config = dict()
        config['n_users'] = data_generator.n_users
        config['n_items'] = data_generator.n_items
        config['n_relations'] = data_generator.n_relations
        config['n_entities'] = data_generator.n_entities

        if self.model_type in ['kgat', 'cfkg']:
            "Load the laplacian matrix."
            config['A_in'] = sum(data_generator.lap_list).tocsr()

            "Load the KG triplets."
            config['all_h_list'] = data_generator.all_h_list
            config['all_r_list'] = data_generator.all_r_list
            config['all_t_list'] = data_generator.all_t_list
            config['all_v_list'] = data_generator.all_v_list

        self.config = config

        self._init_model()

        # TODO Close all sessions used for training and open a new one for the "_best_model"
        # close session tensorflow
        self.sess.close()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        """
            *********************************************************
            Reload the model parameters to fine tune.
            """
        # if self.args.pretrain == 1:
        #     if self.args.model_type in ['bprmf', 'cke', 'fm', 'cfkg']:
        #         pretrain_path = '%sweights/%s/%s/l%s_r%s' % (
        #             self.args.weights_path, self.args.dataset, self.model.model_type, str(self.args.lr),
        #             '-'.join([str(r) for r in eval(self.args.regs)]))
        #
        #     elif self.args.model_type in ['ncf', 'nfm', 'kgat']:
        #         layer = '-'.join([str(l) for l in eval(self.args.layer_size)])
        #         pretrain_path = '%sweights/%s/%s/%s/l%s_r%s' % (
        #             self.args.weights_path, self.args.dataset, self.model.model_type, layer, str(self.args.lr),
        #             '-'.join([str(r) for r in eval(self.args.regs)]))
        #
        #     ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        #     if ckpt and ckpt.model_checkpoint_path:
        #         self.sess.run(tf.global_variables_initializer())
        #         saver.restore(self.sess, ckpt.model_checkpoint_path)
        #         print('load the pretrained model parameters from: ', pretrain_path)
        #
        #         # *********************************************************
        #         # get the performance from the model to fine tune.
        #         if self.args.report != 1:
        #             users_to_test = list(data_generator.test_user_dict.keys())
        #
        #             ret = test(self.sess, self.model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)
        #
        #             pretrain_ret = 'pretrained model recall=[%.5f, %.5f], precision=[%.5f, %.5f], hit=[%.5f, %.5f],' \
        #                            'ndcg=[%.5f, %.5f], auc=[%.5f]' % \
        #                            (ret['recall'][0], ret['recall'][-1],
        #                             ret['precision'][0], ret['precision'][-1],
        #                             ret['hit_ratio'][0], ret['hit_ratio'][-1],
        #                             ret['ndcg'][0], ret['ndcg'][-1], ret['auc'])
        #             print(pretrain_ret)
        #
        #             # *********************************************************
        #             # save the pretrained model parameters of mf (i.e., only user & item embeddings) for pretraining other models.
        #             if self.args.save_flag == -1:
        #                 user_embed, item_embed = self.sess.run(
        #                     [self.model.weights['user_embedding'], self.model.weights['item_embedding']],
        #                     feed_dict={})
        #                 # temp_save_path = '%spretrain/%s/%s/%s_%s.npz' % (args.proj_path, args.dataset, args.model_type, str(args.lr),
        #                 #                                                  '-'.join([str(r) for r in eval(args.regs)]))
        #                 temp_save_path = '%spretrain/%s/%s.npz' % (
        #                     self.args.proj_path, self.args.dataset, self.model.model_type)
        #                 ensureDir(temp_save_path)
        #                 np.savez(temp_save_path, user_embed=user_embed, item_embed=item_embed)
        #                 print('save the weights of fm in path: ', temp_save_path)
        #                 exit()
        #
        #             # *********************************************************
        #             # save the pretrained model parameters of kgat (i.e., user & item & kg embeddings) for pretraining other models.
        #             if self.args.save_flag == -2:
        #                 user_embed, entity_embed, relation_embed = self.sess.run(
        #                     [self.model.weights['user_embed'], self.model.weights['entity_embed'],
        #                      self.model.weights['relation_embed']],
        #                     feed_dict={})
        #
        #                 temp_save_path = '%spretrain/%s/%s.npz' % (
        #                     self.args.proj_path, self.args.dataset, self.args.model_type)
        #                 ensureDir(temp_save_path)
        #                 np.savez(temp_save_path, user_embed=user_embed, entity_embed=entity_embed,
        #                          relation_embed=relation_embed)
        #                 print('save the weights of kgat in path: ', temp_save_path)
        #                 exit()
        #
        #     else:
        #         self.sess.run(tf.global_variables_initializer())
        #         print('without pretraining.')
        # else:
        #     self.sess.run(tf.global_variables_initializer())
        #     print('without pretraining.')

        self.sess.run(tf.global_variables_initializer())
        print('without pretraining.')

        """
            *********************************************************
            Get the final performance w.r.t. different sparsity levels.
            """
        # if self.args.report == 1:
        #     assert self.args.test_flag == 'full'
        #     users_to_test_list, split_state = data_generator.get_sparsity_split()
        #
        #     users_to_test_list.append(list(data_generator.test_user_dict.keys()))
        #     split_state.append('all')
        #
        #     save_path = '%sreport/%s/%s.result' % (self.args.proj_path, self.args.dataset, self.model.model_type)
        #     ensureDir(save_path)
        #     f = open(save_path, 'w')
        #     f.write('embed_size=%d, lr=%.4f, regs=%s, loss_type=%s, \n' % (
        #         self.args.embed_size, self.args.lr, self.args.regs,
        #         self.args.loss_type))
        #
        #     for i, users_to_test in enumerate(users_to_test_list):
        #         ret = test(self.sess, self.model, users_to_test, drop_flag=False, batch_test_flag=batch_test_flag)
        #
        #         final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
        #                      ('\t'.join(['%.5f' % r for r in ret['recall']]),
        #                       '\t'.join(['%.5f' % r for r in ret['precision']]),
        #                       '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
        #                       '\t'.join(['%.5f' % r for r in ret['ndcg']]))
        #         print(final_perf)
        #
        #         f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        #     f.close()
        #     exit()

        ###############################################################################
        ### This is a standard training with early stopping part, most likely you won't need to change it

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
            btime = time()

            batch_data = self.data_generator.generate_train_batch()
            feed_dict = self.data_generator.generate_train_feed_dict(self.model, batch_data)

            _, batch_loss, batch_base_loss, batch_kge_loss, batch_reg_loss = self.model.train(self.sess,
                                                                                              feed_dict=feed_dict)

            loss += batch_loss
            base_loss += batch_base_loss
            kge_loss += batch_kge_loss
            reg_loss += batch_reg_loss

        if np.isnan(loss) == True:
            print('ERROR: loss@phase1 is nan.')
            sys.exit()

        """
        *********************************************************
        Alternative Training for KGAT:
        ... phase 2: to train the KGE method & update the attentive Laplacian matrix.
        """
        if self.model_type in ['kgat']:

            n_A_batch = len(self.data_generator.all_h_list) // self.batch_size_kg + 1

            if self.use_kge is True:
                # using KGE method (knowledge graph embedding).
                for idx in range(n_A_batch):
                    btime = time()

                    A_batch_data = self.data_generator.generate_train_A_batch()
                    feed_dict = self.data_generator.generate_train_A_feed_dict(self.model, A_batch_data)

                    _, batch_loss, batch_kge_loss, batch_reg_loss = self.model.train_A(self.sess, feed_dict=feed_dict)

                    loss += batch_loss
                    kge_loss += batch_kge_loss
                    reg_loss += batch_reg_loss

            if self.use_att is True:
                # updating attentive laplacian matrix.
                self.model.update_attentive_A(self.sess)

        if np.isnan(loss) == True:
            print('ERROR: loss@phase2 is nan.')
            sys.exit()

        perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f]' % (
            currentEpoch, time() - t1, loss, base_loss, kge_loss, reg_loss)
        print(perf_str)

    def save_model(self, folder_path, file_name=None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        # TODO replace this with the Saver required by the model
        #  in this case the neural network will be saved with the _weights suffix, which is rather standard
        # self.model.save_weights(folder_path + file_name + "_weights", overwrite=True)
        # layer = '-'.join([str(l) for l in eval(self.args.layer_size)])
        # weights_save_path = '%s/weights/%s/%s/%s/l%s_r%s' % (
        #     self.args.weights_path, self.args.dataset, self.model.model_type, layer, str(self.args.lr),
        #     '-'.join([str(r) for r in eval(self.args.regs)]))

        # TODO Alternativley you may save the tensorflow model with a session
        saver = tf.train.Saver()
        saver.save(self.sess, folder_path + file_name + "_session")

        # print('save the session in path: ', weights_save_path)

        data_dict_to_save = {
            # TODO replace this with the hyperparameters and attribute list you need to re-instantiate
            #  the model when calling the load_model

        }

        # Do not change this
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

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

        # TODO replace this with what required to re-instantiate the model and load its weights,
        #  Call the init_model function you created before
        self._init_model()
        # self.model.load_weights(folder_path + file_name + "_weights")

        # TODO If you are using tensorflow, you may instantiate a new session here
        # TODO reset the default graph to "clean" the tensorflow state
        tf.reset_default_graph()
        weights = tf.get_variable("weights", [3, 2])
        saver = tf.train.Saver()
        saver.restore(self.sess, folder_path + file_name + "_session")

        self._print("Loading complete")
