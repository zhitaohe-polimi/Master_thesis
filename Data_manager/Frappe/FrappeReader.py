#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19/02/2019

@author: Simone Boglio
"""



import zipfile, shutil
import pandas as pd
from Data_manager.DatasetMapperManager import DatasetMapperManager
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL, remove_Dataframe_duplicates


class FrappeReader(DataReader):
    """
    Reference for the dataset:
    https://www.baltrunas.info/context-aware

    @article{DBLP:journals/corr/BaltrunasCKO15,
      author    = {Linas Baltrunas and
                   Karen Church and
                   Alexandros Karatzoglou and
                   Nuria Oliver},
      title     = {Frappe: Understanding the Usage and Perception of Mobile App Recommendations
                   In-The-Wild},
      journal   = {CoRR},
      volume    = {abs/1505.03014},
      year      = {2015},
      url       = {http://arxiv.org/abs/1505.03014},
      eprinttype = {arXiv},
      eprint    = {1505.03014},
      timestamp = {Mon, 13 Aug 2018 16:47:33 +0200},
      biburl    = {https://dblp.org/rec/journals/corr/BaltrunasCKO15.bib},
      bibsource = {dblp computer science bibliography, https://dblp.org}
    }
    """
    DATASET_URL = "https://github.com/hexiangnan/neural_factorization_machine/archive/master.zip"
    DATASET_SUBFOLDER = "Frappe/"
    AVAILABLE_URM = ["URM_all", "URM_occurrence"]
    AVAILABLE_ICM = []
    AVAILABLE_UCM = []
    DATASET_SPECIFIC_MAPPER = []


    def __init__(self):
        super(FrappeReader, self).__init__()


    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER



    def _load_from_original_file(self):
        # Load data from original

        self._print("Loading original data")

        zipFile_path = self.DATASET_SPLIT_ROOT_FOLDER + self.DATASET_SUBFOLDER

        try:

            dataFile = zipfile.ZipFile(zipFile_path + "neural_factorization_machine-master.zip")

        except (FileNotFoundError, zipfile.BadZipFile):

            self._print("Unable to find data zip file. Downloading...")

            download_from_URL(self.DATASET_URL, zipFile_path, "neural_factorization_machine-master.zip")

            dataFile = zipfile.ZipFile(zipFile_path + "neural_factorization_machine-master.zip")



        inner_path_in_zip = "neural_factorization_machine-master/data/frappe/"


        URM_train_path = dataFile.extract(inner_path_in_zip + "frappe.train.libfm", path=zipFile_path + "decompressed/")
        URM_test_path = dataFile.extract(inner_path_in_zip + "frappe.test.libfm", path=zipFile_path + "decompressed/")
        URM_validation_path = dataFile.extract(inner_path_in_zip + "frappe.validation.libfm", path=zipFile_path + "decompressed/")


        URM_all_dataframe = pd.concat([self._loadURM(URM_train_path),
                                       self._loadURM(URM_test_path),
                                       self._loadURM(URM_validation_path)])

        URM_occurrence_dataframe = URM_all_dataframe.groupby(["UserID","ItemID"],as_index=False)["Data"].sum()

        URM_all_dataframe = remove_Dataframe_duplicates(URM_all_dataframe,
                                                        unique_values_in_columns = ['UserID', 'ItemID'],
                                                        keep_highest_value_in_col = "Data")



        dataset_manager = DatasetMapperManager()
        dataset_manager.add_URM(URM_all_dataframe, "URM_all")
        dataset_manager.add_URM(URM_occurrence_dataframe, "URM_occurrence")

        loaded_dataset = dataset_manager.generate_Dataset(dataset_name=self._get_dataset_name(),
                                                          is_implicit=self.IS_IMPLICIT)



        self._print("Cleaning Temporary Files")

        shutil.rmtree(zipFile_path + "decompressed", ignore_errors=True)

        self._print("Loading Complete")

        return loaded_dataset





    def _loadURM(self, file_name, header = False, separator = " "):


        fileHandle = open(file_name, "r")

        if header:
            fileHandle.readline()

        item_list = []
        user_list = []

        for index, line in enumerate(fileHandle):

            if (index % 100000 == 0 and index!=0):
                print("Processed {} rows".format(index))

            line = line.split(separator)
            if (len(line)) > 1:
                if line[0]=='1':
                    item_list.append(line[2].split(':')[0])
                    user_list.append(line[1].split(':')[0])

                elif line[0]=='-1':
                    pass
                else:
                    print('ERROR READING DATASET')
                    break

        fileHandle.close()

        URM_dataframe = pd.DataFrame({"UserID": user_list,
                                      "ItemID": item_list,
                                      "Data":  [1]*len(user_list),
                                      })

        return  URM_dataframe


