import pandas as pd
import os

from Conferences.HGB.HGB_our_interface.DatasetProvided.MultiDatasetsReader import MultiDatasetsReader
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender


if __name__ == '__main__':
    dataset_name = "movie-lens"
    dataset_path = '/home/ubuntu/Master_thesis/Conferences/HGB/HGB_github/baseline/Data/'

    if dataset_name == "movie-lens" or dataset_name == "last-fm" or dataset_name == "yelp2018" or dataset_name == "amazon-book":
        dataset = MultiDatasetsReader(dataset_path + dataset_name)

    URM_train = dataset.URM_DICT["URM_train"].copy()
    URM_validation = dataset.URM_DICT["URM_validation"].copy()
    URM_test = dataset.URM_DICT["URM_test"].copy()

    URM_submission_train = URM_train+URM_validation

    rec = ItemKNNCFRecommender(URM_submission_train)
    rec.load_model(
        folder_path='result_experiments/baseline/{}/models_Recall'.format(dataset_name),
        file_name='ItemKNNCFRecommender_cosine_best_model_last.zip')
    #UserKNNCFRecommender_jaccard

    # ######## ignore out of stock
    # df_articles = pd.read_parquet('{}/processed_articles.parquet'.format(DATASET_PATH))
    # df_article_out_of_stock = df_articles.query("out_of_stock==1")[
    #     ['article_id', 'out_of_stock']]  # ['article_id'].unique().tolist()
    #
    # item_original_ID_to_index_mapper = dataset.get_item_original_ID_to_index_mapper()
    # df_article_out_of_stock['article_id_index'] = df_article_out_of_stock.apply(
    #     lambda x: item_original_ID_to_index_mapper[x.article_id], axis=1)
    #
    # print(df_article_out_of_stock)
    #
    # out_of_stock_list = df_article_out_of_stock['article_id_index'].unique().tolist()
    # recommender.set_items_to_ignore(out_of_stock_list)
    #
    # #####################

    path = "result_experiments/{}/{}/".format('baseline', dataset_name)

    save_path = os.path.join(path, "{}-recommendation_results.csv".format(dataset_name+"_"+rec.RECOMMENDER_NAME))

    f = open(save_path, "w")
    f.write("customer_id,prediction\n")

    for i in range(URM_submission_train.shape[0]):
        recommended_items = rec.recommend(i, cutoff=20, remove_seen_flag=False)
        well_formatted = str(i)+","+" ".join([str(x) for x in recommended_items])
        f.write(f"{i}, {well_formatted}\n")
        print("%s:%s" % (i, well_formatted))
    f.close()
    print("save complete")
