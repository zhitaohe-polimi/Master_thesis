import dgl
import numpy as np
import unittest, shutil

import torch

from Conferences.HGB.HGB_github.baseline.Model.utility.loader_kgat import KGAT_loader
from Conferences.HGB.HGB_our_interface.DatasetProvided.MultiDatasetsReader import MultiDatasetsReader
from Conferences.HGB.HGB_our_interface.baseline_RecommenderWrapper import baseline_RecommenderWrapper


class MyTestCase(unittest.TestCase):
    def test1(self):
        print("this is test1")

    def test2(self):
        recommender_class = baseline_RecommenderWrapper  # TODO ADD the class of the recommender you ported

        data_path = "D:/Polimi/M2-S2/Thesis/RecSys_Porting_Project-master/Conferences/HGB/HGB_github/baseline/Data/"
        dataset_name = "movie-lens"

        recommender_instance, URM_train, URM_test = get_data_and_rec_instance(recommender_class, MultiDatasetsReader,
                                                                              data_path + dataset_name)

    def test_compute_item_score(self):
        proj_path = "D:/Polimi/M2-S2/Thesis/RecSys_Porting_Project-master/Conferences/HGB/HGB_github/baseline/Model/"
        data_path = "D:/Polimi/M2-S2/Thesis/RecSys_Porting_Project-master/Conferences/HGB/HGB_github/baseline/Data/"
        dataset_name = "movie-lens"

        """
        *********************************************************
        Use the pretrained data to initialize the embeddings.
        """
        recommender_class = baseline_RecommenderWrapper  # TODO ADD the class of the recommender you ported

        recommender_instance, URM_train, URM_test = get_data_and_rec_instance(recommender_class, MultiDatasetsReader,
                                                                              data_path + dataset_name)

        n_users, n_items = URM_train.shape

        pretrain_data = load_pretrained_data(proj_path, dataset_name, pretrain=-1)

        baseline_hyperparameters = {
            "pretrain": pretrain_data,
            "dataset": dataset_name,
            "data_path": data_path,
        }

        recommender_instance.fit(epochs=10, **baseline_hyperparameters)

        user_batch_size = 1000

        user_id_list = np.arange(n_users, dtype=np.int)
        np.random.shuffle(user_id_list)
        user_id_list = user_id_list[:user_batch_size]

        item_scores = recommender_instance._compute_item_score(user_id_array=user_id_list,
                                                               items_to_compute=None)

        user_batch_size = min(user_batch_size, n_users)

        self.assertEqual(item_scores.shape, (user_batch_size, n_items),
                         "item_scores shape not correct, contains more users than in user_id_array")
        self.assertFalse(np.any(np.isnan(item_scores)), "item_scores contains np.nan values")

        item_batch_size = 500

        item_id_list = np.arange(n_items, dtype=np.int)
        np.random.shuffle(item_id_list)
        item_id_list = item_id_list[:item_batch_size]

        item_scores = recommender_instance._compute_item_score(user_id_array=user_id_list,
                                                               items_to_compute=item_id_list)

        self.assertEqual(item_scores.shape, (user_batch_size, n_items),
                         "item_scores shape not correct, does not contain all items")
        self.assertFalse(np.any(np.isnan(item_scores)), "item_scores contains np.nan values")
        self.assertFalse(np.any(np.isposinf(item_scores)), "item_scores contains +np.inf values")

        # Check items not in list have a score of -np.inf
        item_id_not_to_compute = np.ones(n_items, dtype=np.bool)
        item_id_not_to_compute[item_id_list] = False

        # print(item_scores[:, item_id_not_to_compute],np.isneginf(item_scores[:, item_id_not_to_compute]),
        #       np.all(np.isneginf(item_scores[:, item_id_not_to_compute])),
        #              np.any(np.isneginf(item_scores[:, item_id_not_to_compute])))

        # ?? original code=assertTrue
        self.assertFalse(np.all(np.isneginf(item_scores[:, item_id_not_to_compute])),
                         "item_scores contains scores for items that should not be computed")

    def test_save_and_load(self):
        proj_path = "D:/Polimi/M2-S2/Thesis/RecSys_Porting_Project-master/Conferences/HGB/HGB_github/baseline/Model/"
        data_path = "D:/Polimi/M2-S2/Thesis/RecSys_Porting_Project-master/Conferences/HGB/HGB_github/baseline/Data/"
        dataset_name = "movie-lens"

        recommender_class = baseline_RecommenderWrapper  # TODO ADD the class of the recommender you ported

        recommender_instance, URM_train, URM_test = get_data_and_rec_instance(recommender_class, MultiDatasetsReader,
                                                                              data_path + dataset_name)
        n_users, n_items = URM_train.shape

        pretrain_data = load_pretrained_data(proj_path, dataset_name, pretrain=-1)

        baseline_hyperparameters = {
            "pretrain": pretrain_data,
            "dataset": dataset_name,
            "data_path": data_path,
        }


        folder_path = "./temp_folder/"
        file_name = "temp_file"

        recommender_instance.fit(epochs=10, **baseline_hyperparameters)
        recommender_instance.save_model(folder_path=folder_path, file_name=file_name)

        recommender_instance_loaded = recommender_class(URM_train)
        recommender_instance_loaded.load_model(folder_path=folder_path, file_name=file_name)


        # print("Result original: {}\n".format(results_run_original))
        # print("Result loaded: {}\n".format(results_run_loaded))

        for user_id in range(n_users):

            item_scores_original = recommender_instance._compute_item_score(user_id_array=[user_id],
                                                                                     items_to_compute=None)

            item_scores_original_2 = recommender_instance._compute_item_score(user_id_array=[user_id],
                                                                                       items_to_compute=None)

            is_stochastic = not np.allclose(item_scores_original, item_scores_original_2)


            if not is_stochastic:

                item_scores_loaded = recommender_instance_loaded._compute_item_score(user_id_array=[user_id],
                                                                                     items_to_compute=None)

                self.assertTrue(np.allclose(item_scores_original, item_scores_loaded),
                                "item_scores of the fitted model and of the loaded model are different")

            else:
                print(
                    "Algorithm is stochastic, since scores will be different between runs the integrity of the loaded model cannot be verified based on it computing the same predictions")
                break

        shutil.rmtree(folder_path, ignore_errors=True)


def load_pretrained_data(proj_path, dataset, pretrain):
    pre_model = 'mf'
    if pretrain == -2:
        pre_model = 'kgat'
    pretrain_path = '%spretrain/%s/%s.npz' % (proj_path, dataset, pre_model)
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained bprmf model parameters.')
    except Exception:
        pretrain_data = None
    return pretrain_data


def get_data_and_rec_instance(recommender_class, dataset_class, path):
    dataset_object = dataset_class(path)

    URM_train, URM_test = dataset_object.URM_DICT["URM_train"].copy(), dataset_object.URM_DICT["URM_test"].copy()

    recommender_instance = recommender_class(URM_train)

    return recommender_instance, URM_train, URM_test


if __name__ == '__main__':
    unittest.main()
