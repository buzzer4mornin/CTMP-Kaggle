#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import shutil
import sys
import time
import pickle
import numpy as np
import pandas as pd
from math import floor

from CTMP import MyCTMP
from LDA import MyLDA

sys.path.insert(0, './CTMP/common/')
import utilities


# ------------ RUN in terminal ------------
# --> python ./CTMP/model/run_model.py

def main():
    # Get environment variables
    which_model = "ctmp"
    which_size = "original"
    k_cross_val = 5

    docs_file = "./CTMP/input-data/docs.txt"
    rating_file = "./CTMP/input-data/df_rating_UPDATED"
    setting_file = "./CTMP/input-data/settings.txt"
    output_folder = "../working/"

    # -------------------------------------------- Get Data --------------------------------------------------------
    # Read & write settings into model folder
    print('reading setting ...')
    ddict = utilities.read_setting(setting_file)
    print('write setting ...')
    file_name = f'{output_folder}setting.txt'

    os.chdir("..")
    utilities.write_setting(ddict, file_name)
    os.chdir("./wdirrrr/")

    wordids, wordcts = utilities.read_data(docs_file)

    rating_GroupForUser_train = pickle.load(open("./CTMP/input-data/rating_GroupForUser_train.pkl", "rb"))
    rating_GroupForMovie_train = pickle.load(open("./CTMP/input-data/rating_GroupForMovie_train.pkl", "rb"))
    # rating_GroupForUser_test = pickle.load(open("./CTMP/input-data/rating_GroupForUser_test.pkl", "rb"))
    # rating_GroupForMovie_test = pickle.load(open("./CTMP/input-data/rating_GroupForMovie_test.pkl", "rb"))

    # -------------------------------------- Initialize Algorithm --------------------------------------------------
    if which_model == "ctmp":
        print('initializing CTMP algorithm ...\n')
        algo = MyCTMP(rating_GroupForUser_train, rating_GroupForMovie_train,
                      ddict['num_docs'], ddict['num_terms'], ddict['num_topics'],
                      ddict["user_size"], ddict["lamb"], ddict["e"], ddict["f"], ddict['alpha'],
                      ddict['iter_infer'])

    else:
        print('initializing LDA algorithm ...\n')
        algo = MyLDA(ddict['num_docs'], ddict['num_terms'], ddict['num_topics'], ddict['alpha'],
                     ddict['iter_infer'])

    # ----------------------------------------- Run Algorithm ------------------------------------------------------
    print('START!')

    for i in range(ddict['iter_train']):
        print(f'\n*** iteration: {i} ***\n')
        time.sleep(2)
        # run single EM step and return attributes
        algo.run_EM(wordids, wordcts, i)



    print('DONE!')

    # ----------------------------------------- Write Results ------------------------------------------------------
    # Search top words of each topics
    list_tops = utilities.list_top(algo.beta, ddict['tops'])

    print("\nsaving the final results.. please wait..")
    os.chdir("..")
    output_folder = "../working"
    utilities.write_file(output_folder, list_tops, algo)


if __name__ == '__main__':
    main()
