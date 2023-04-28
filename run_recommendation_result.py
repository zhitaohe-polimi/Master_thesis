import pandas as pd
import os

from Conferences.HGB.HGB_our_interface.DatasetProvided.MultiDatasetsReader import MultiDatasetsReader
from Conferences.HGB.HGB_our_interface.baseline_RecommenderWrapper import baseline_RecommenderWrapper
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
# from Recommenders.MatrixFactorization.Cython.new_algo_with_MFAttention_Cython import \
#     new_MatrixFactorization_FunkSVD_Cython
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython, \
    MatrixFactorization_BPR_Cython, MatrixFactorization_AsySVD_Cython
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.MatrixFactorization.PyTorchNewMF import PyTorchNewMF_BPR_Recommender
from Recommenders.MatrixFactorization.PyTorchMF import PyTorchMF_MSE_Recommender, PyTorchMF_BPR_Recommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

if __name__ == '__main__':
    dataset_name = "movie-lens"
    dataset_path = '/home/tesista/Master_thesis/Conferences/HGB/HGB_github/baseline/Data/'

    if dataset_name == "movie-lens" or dataset_name == "last-fm" or dataset_name == "yelp2018" or dataset_name == "amazon-book":
        dataset = MultiDatasetsReader(dataset_path + dataset_name)

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()

    URM_submission_train = URM_train + URM_validation

    # recommender_class_list = [
    #     # P3alphaRecommender,
    #     # SLIM_BPR_Cython,
    #     # MatrixFactorization_BPR_Cython,
    #     # IALSRecommender,
    #     # MatrixFactorization_FunkSVD_Cython,
    #     MatrixFactorization_AsySVD_Cython,
    #     # ItemKNNCFRecommender,
    #     # UserKNNCFRecommender,
    #     # PureSVDRecommender,
    #     # PyTorchNewMF_BPR_Recommender,
    #     # PyTorchMF_BPR_Recommender,
    # ]
    #
    # for rec in recommender_class_list:
    #     try:
    #         rec = rec(URM_submission_train)
    #         rec.load_model(
    #             folder_path='result_experiments/baseline/{}/models_RECALL/'.format(dataset_name),
    #             file_name='{}_best_model.zip'.format(rec.RECOMMENDER_NAME))
    # #UserKNNCFRecommender_jaccard
    #
    # # rec = UserKNNCFRecommender(URM_submission_train)
    # # rec.load_model(
    # #     folder_path='result_experiments/baseline/{}/models_RECALL/'.format(dataset_name),
    # #     file_name='UserKNNCFRecommender_jaccard_best_model_last.zip')
    #
    # # ######## ignore out of stock
    # # df_articles = pd.read_parquet('{}/processed_articles.parquet'.format(DATASET_PATH))
    # # df_article_out_of_stock = df_articles.query("out_of_stock==1")[
    # #     ['article_id', 'out_of_stock']]  # ['article_id'].unique().tolist()
    # #
    # # item_original_ID_to_index_mapper = dataset.get_item_original_ID_to_index_mapper()
    # # df_article_out_of_stock['article_id_index'] = df_article_out_of_stock.apply(
    # #     lambda x: item_original_ID_to_index_mapper[x.article_id], axis=1)
    # #
    # # print(df_article_out_of_stock)
    # #
    # # out_of_stock_list = df_article_out_of_stock['article_id_index'].unique().tolist()
    # # recommender.set_items_to_ignore(out_of_stock_list)
    # #
    # # #####################
    #
    #         path = "result_experiments/{}/{}/".format('baseline', dataset_name)
    #
    #         save_path = os.path.join(path, "{}-recommendation_results.csv".format(dataset_name+"_"+rec.RECOMMENDER_NAME))
    #
    #         f = open(save_path, "w")
    #         f.write("customer_id,prediction\n")
    #
    #         for i in range(URM_submission_train.shape[0]):
    #             recommended_items = rec.recommend(i, cutoff=20, remove_seen_flag=False)
    #             well_formatted = str(i)+","+" ".join([str(x) for x in recommended_items])
    #             f.write(f"{i}, {well_formatted}\n")
    #             print("%s:%s" % (i, well_formatted))
    #         f.close()
    #         print("save complete")
    #     except Exception as e:
    #         print(e)

    # rec = ItemKNNCFRecommender(URM_submission_train)
    # rec.load_model(
    #     folder_path='result_experiments/baseline/{}/models_RECALL/'.format(dataset_name),
    #     file_name='ItemKNNCFRecommender_cosine_best_model_last.zip')

    # rec = baseline_RecommenderWrapper(URM_submission_train)
    # rec.load_model(
    #     folder_path='result_experiments/baseline/{}/'.format(dataset_name+'_4'),
    #     file_name='_best_model')

    rec = PyTorchNewMF_BPR_Recommender(URM_submission_train)
    rec.load_model(
            folder_path='result_experiments/baseline/{}/'.format(dataset_name),
            file_name='_best_model.zip')

    path = "result_experiments/{}/{}/".format('baseline', dataset_name)

    save_path = os.path.join(path, "{}-recommendation_results.csv".format(dataset_name + "_" + rec.RECOMMENDER_NAME))

    f = open(save_path, "w")
    f.write("customer_id,prediction\n")

    for i in range(URM_submission_train.shape[0]):
        recommended_items = rec.recommend(i, cutoff=20, remove_seen_flag=False)
        well_formatted = str(i) + "," + " ".join([str(x) for x in recommended_items])
        f.write(f"{i}, {well_formatted}\n")
        print("%s:%s" % (i, well_formatted))
    f.close()
    print("save complete")

