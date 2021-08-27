# !/usr/bin/env python

# /***********************************************************************
#
# 	This file is part of KEEL-software, the Data Mining tool for regression,
# 	classification, clustering, pattern mining and so on.
#
# 	Copyright (C) 2004-2010
#
# 	F. Herrera (herrera@decsai.ugr.es)

#
# 	This program is free software: you can redistribute it and/or modify
# 	it under the terms of the GNU General Public License as published by
# 	the Free Software Foundation, either version 3 of the License, or
# 	(at your option) any later version.
#
# 	This program is distributed in the hope that it will be useful,
# 	but WITHOUT ANY WARRANTY; without even the implied warranty of
# 	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# 	GNU General Public License for more details.
#
# 	You should have received a copy of the GNU General Public License
# 	along with this program.  If not, see http://www.gnu.org/licenses/
#
# **********************************************************************
# 

import numpy as np

# * <p>It reads the configuration file (data-set files and parameters) and launch the algorithm</p>
# *
# * @version 1.0
# * @since JDK1.5
from LoadFiles import LoadFiles
from FarcHDClassifier import FarcHDClassifier
from Logger import Logger
import os
from pathlib import Path


class Main:
    # * Main Program
    # * @param args It contains the name of the configuration file
    # * Format:
    # * algorithm = ;algorithm name>
    # * inputData = "training file" "validation file" "test file"
    # * outputData = "training file" "test file"
    # *
    # * seed = value (if used)
    # Parameter1; value1
    # Parameter2&gt; value2

    if __name__ == "__main__":
        # print("Executing Algorithm.")

        # print("sys.argv: " + sys.argv[1])
        # execute(sys.argv[1])

        logger = Logger.set_logger()
        lf = LoadFiles()

        log_file = open("help.log", "w")
        log_file.truncate()
        log_file.close()
        # logger.debug("Begin  lf.parse_configuration_file in Main ")

        dataset_folder = 'A3'
        config_folder = 'config'
        config_file = "config0s0.txt"
        # whole_file_name_with_path = os.getcwd() + config_file

        # lf.parse_configuration_file("\iris", "config1s0.txt")

        whole_file_name_with_path = os.path.join(os.getcwd(), config_file)
        cwd = Path.cwd()
        whole_file_name_with_path = cwd / dataset_folder / config_folder / config_file
        lf.parse_configuration_file(whole_file_name_with_path, dataset_folder)


        # logger.debug("Begin  FarcHDClassifier in Main ")
        farchd_classifier = FarcHDClassifier(lf)

        X = farchd_classifier.get_X()
        y = farchd_classifier.get_y()
        indices = np.random.permutation(len(X))

        X_test = farchd_classifier.get_test_x()
        y_test = farchd_classifier.get_test_y()

        # logger.debug("Begin  farchd_classifier.fit in Main ")
        y.reshape(1, -1)
        farchd_classifier.fit(X, y)
        test_x = [4.6, 3.1, 1.5, 0.2]

        if_granularity = False

        # normal rule prediction
        if_train = True
        predict_y_train = farchd_classifier.predict(X)

        # logger.debug("Begin  farchd_classifier.score Main ")

        farchd_classifier.score(y,predict_y_train, if_granularity, if_train)

        if_train = False
        predict_y_test = farchd_classifier.predict(X_test)
        farchd_classifier.score(y_test, predict_y_test, if_granularity, if_train)


        """ 

        if_granularity = True

        # granularity  rule for train data prediction
        if_train = True
        predict_train_granularity_y = farchd_classifier.predict_granularity(X)
        farchd_classifier.score(y, predict_train_granularity_y, if_granularity, if_train)

        # granularity  rule for test data prediction
        if_train = False
        predict_test_granularity_y = farchd_classifier.predict_granularity(X_test)
        farchd_classifier.score(y_test, predict_test_granularity_y,if_granularity, if_train)
        """

