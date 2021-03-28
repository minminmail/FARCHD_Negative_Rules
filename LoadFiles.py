"""
This file is for prepare the config file and read training file or test file , to get parameters informatoin and data set informaton.

@ author Written by Rui Min
@ version 1.0
@ Python 3

"""


import logging
from MyDataSet import MyDataSet
import numpy as np
from pathlib import Path


class LoadFiles:
    algorithm_name = ""
    training_file = ""
    validation_file = ""
    test_file = ""

    input_files = []
    output_tr_file = ""
    output_ts_file = ""
    output_files = []
    file_path = None
    result_path = None
    data_folder = None

    train_mydataset = None
    val_mydataset = None
    test_mydataset = None
    file_to_open=None
    something_wrong= None

    parameters = []

    def __init__(self):
        self.input_files = []
        self.output_files = []
        self.parameters = []
        self.file_path = None
        self.result_path = None
        self.data_folder = None
        self.something_wrong = False

    def parse_configuration_file(self, file_name,dataset_folder):

        self.file_to_open = file_name

        with file_name.open() as config_file:
            line = config_file.readlines()
            print(line)

        #config_file = open(self.file_to_open, "r")

        #logging.info("fileName in parseParameters = " + file_name)
        #logging.info("before open file")

        db_path = "dataset"
        self.file_path = Path(dataset_folder)/db_path
        self.result_path = dataset_folder


        # file is an string containing the whole file
        #file_string = config_file.read()
        #line = file_string.splitlines()

        for line_number in range(0, len(line)):
            # print("In line " + str(lineNumber) + ", the str is:begin ***   " + line[lineNumber] + "   ***end")
            if line_number == 0:
                self.read_name(line[line_number])
            elif line_number == 1:
                self.read_input_files(line[line_number])  # We read all the input files
            elif line_number == 2:
                self.read_output_files(line[line_number])  # We read all the output files
            else:  # read parameters and save into map
                self.read_all_parameters(line[line_number])  # We read all the possible parameters
        # print("__init__ of Read Files begin...")

        self.train_mydataset = MyDataSet()
        self.val_mydataset = MyDataSet()
        self.test_mydataset = MyDataSet()

        try:

            input_training_file = self.get_input_training_files()
            print("Reading the training set: " + input_training_file)

        

            self.train_mydataset.read_classification_set(input_training_file, True, self.file_path)
            
            print("Reading the validation set: ")
            input_validation_file = self.get_validation_input_file()
        
            self.val_mydataset.read_classification_set(input_validation_file, True, self.file_path)

            print("Reading the test set: ")
            input_test_file = self.get_input_test_files()
      
            self.test_mydataset.read_classification_set(input_test_file, False, self.file_path)
            print(" ********* test_mydataset.myDataSet read_classification_set finished !!!!!! *********")
        except IOError as ioError:
            print("I/O error: " + str(ioError))
            self.something_wrong = True
        except Exception as e:
            print("Unexpected error:" + str(e))
            self.something_wrong = True

        self.something_wrong = self.something_wrong or self.train_mydataset.has_missing_attributes()
        # print("********* Summary for readAllParameters :" + " *********")
        # for key, value in self.__parameters:
        # print("********* parameters are : (" + key + ", " + value + " ) *********")

    # """
    #     * It reads the name of the algorithm from the configuration file
    #     * @param line StringTokenizer It is the line containing the algorithm name.
    # """
    def read_name(self, line):
        # print("In side the readName method the parameter pass is :" + str(line))
        name = line.rpartition("=")[2]
        name = name.strip()
        # print("In side the readName method after split =, we get:" + str(name))
        self.algorithm_name = name

    # """
    #     * We read the input data-set files and all the possible input files
    #     * @param line StringTokenizer It is the line containing the input files.
    # """
    def read_input_files(self, line):
        # print("Inside the readInputFiles mehtod, we get parameter is:" + str(line))
        first_parts = line.split()
        line_number = len(first_parts)
        file_list = []
        for i in range(0, line_number):
            whole_name = first_parts[i]
            # print("Inside readInputFiles, line " + str(lineNumber) + ",wholeName: " + str(wholeName))
            file_name_with_str = whole_name.rpartition('/')[2]
            # print("Inside readInputFiles, line " + str(fileNameWithStr) + ",fileNameWithStr: " + str(fileNameWithStr))
            file_name = file_name_with_str[:-1]
            # print("Inside readInputFiles, line " + str(lineNumber) + ",fileName: " + str(fileName))

            file_type = file_name[-3:]
            if file_type == "dat" or file_type == "tra" or file_type == "tst":
                file_list.append(file_name)

        file_number = len(file_list)
        # print("file_number :" + str(file_number))
        for i in range(0, file_number):
            if i == 0:
                self.training_file = file_list[i]
            elif i == 1:
                self.validation_file = file_list[i]
            elif i == 2:
                self.test_file = file_list[i]
            else:
                self.input_files.append(file_list[i])

        # print("The other remaining Input files number is :" + str(len(self.input_files)))

        # for file in self.input_files:
        # print("input file is :" + file)

        # print("********* Summary for readInputFiles :" + " *********")
        # print("********* The Input training file  is :" + str(self.training_file) + " *********")
        # print("********* The Input validation file  is :" + str(self.validation_file) + " *********")
        # print("********* The Input test file  is :" + str(self.test_file) + " *********")

    # """
    #     * We read the output files for training and test and all the possible remaining output files
    #     * @param line StringTokenizer It is the line containing the output files.
    # """
    def read_output_files(self, line):
        # print("Inside the readInputFiles method, we get parameter is:" + str(line))
        first_parts = line.split()
        file_list = []
        line_number = len(first_parts)
        for i in range(0, line_number):
            whole_name = first_parts[i]
            # print("Inside readOutputFiles, line " + str(lineNumber) + ",wholeName: " + str(wholeName))
            file_name_with_str = whole_name.rpartition('/')[2]
            # print("Inside readOutputFiles, line " + str(fileNameWithStr) + ",fileNameWithStr: " + str(fileNameWithStr))
            file_name = file_name_with_str[:-1]
            # print("Inside readOutputFiles, line " + str(lineNumber) + ",fileName: " + str(fileName))

            file_type = file_name[-3:]
            if file_type == "txt" or file_type == "tra" or file_type == "tst":
                file_list.append(file_name)

        file_number = len(file_list)
        # print("file_number" + str(file_number))
        for i in range(0, file_number):
            if i == 0:
                self.output_tr_file = file_list[i]
            elif i == 1:
                self.output_ts_file = file_list[i]
            else:
                self.output_files.append(file_list[i])

        # print("********* Summary for readOutputFiles :" + " *********")
        # print("*********  The output training file  is :" + str(self.__outputTrFile) + " *********")
        # print("*********  The output test file  is :" + str(self.__outputTstFile) + " *********")

        for file in self.output_files:
            print("********* output file is :" + file + " *********")

    # """
    #     * We read all the possible parameters of the algorithm
    #     * @param line StringTokenizer It contains all the parameters.
    # """
    def read_all_parameters(self, line):

        # print("readAllParameters begin,  line is :" + line)
        key = line.rpartition("=")[0]
        # print("The parameter key is :" + key)
        value = line.rpartition("=")[2]
        # print("The parameter value is :" + value)
        # remove the space in key and value of parameters and save into dictionary
        if key != "":
            self.parameters.append((key, value))

    # If the algorithm is non-deterministic the first parameter is the Random SEED

    # """
    # * It returns the algorithm name
    # *
    # * @return the algorithm name
    # """

    def get_algorithm_name(self):
        return self.algorithm_name

    # """
    # * It returns the name of the parameters
    # *
    # * @return the name of the parameters
    # """
    def get_parameters(self):
        param = self.parameters

        return param

    # """
    # * It returns the name of the parameter specified
    # *
    # * @param key the index of the parameter
    # * @return the value of the parameter specified
    # """
    def get_parameter(self, pos):

        return self.parameters[pos][1]

    # """
    # * It returns the input files
    # *
    # * @return the input files
    # """

    def get_input_files(self):
        return str(self.input_files)

    # """
    # * It returns the input training files
    # *
    # * @return the input training files
    # """

    def get_input_training_files(self):
        return self.training_file

    # """
    #  * It returns the input test files
    #  *
    #  * @return the input test files
    #  """

    def get_input_test_files(self):
        return self.test_file

    # * It returns the validation input file
    # *
    # * @return the validation input file

    def get_validation_input_file(self):
        return self.validation_file

    # /**
    #  * It returns the training output file
    #  *
    #  * @return the training output file
    #  */
    def get_training_output_file(self):
        return self.output_tr_file

    # * It returns the test output file
    # *
    # * @return the test output file

    def get_test_output_file(self):
        return self.output_ts_file

    # * It returns the algorithm name
    # *
    # * @return the algorithm name

    def get_algorithm_name(self):
        return self.algorithm_name

    # * It returns the name of the parameters
    # *
    # * @return the name of the parameters

    def get_parameters(self):
        return self.parameters

    # * It returns the input files
    # *
    # * @return the input files

    def get_input_files(self):
        return self.input_files

    # """
    # * It returns the input file of the specified index
    # *
    # * @param pos index of the file
    # * @return the input file of the specified index
    # """
    def get_input_file(self, pos):
        return self.input_files[pos]

    # """
    # * It returns the output files
    # *
    # * @return the output files
    # """
    def get_output_files(self):
        return self.output_files

    # """
    # * It returns the output file of the specified index
    # *
    # * @param pos index of the file
    # * @return the output file of the specified index
    # """
    def get_output_file(self, pos):
        return self.output_files[pos]

    def get_X(self):

        self.X = self.train_mydataset.get_X()
        # change into ndarray type
        self.X = np.array(self.X)
        print(self.X)

        return self.X

    def get_y(self):

        self.y = self.train_mydataset.get_y()
        self.y = np.array(self.y)
        print(self.y)

        return self.y

    def get_test_x(self):

        self.X = self.test_mydataset.get_X()
        # change into ndarray type
        self.X = np.array(self.X)
        print(self.X)

        return self.X

    def get_test_y(self):

        self.y = self.test_mydataset.get_y()
        self.y = np.array(self.y)
        print(self.y)

        return self.y
