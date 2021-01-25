"""
This is a module to be used as a reference for building other modules
"""
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from Apriori import Apriori
from DataBase import DataBase
from MyDataSet import MyDataSet
from GranularityRule import GranularityRule
import datetime
import random
import os
import time

from Populate import Populate
from RuleBase import RuleBase
import numpy as np


class FarcHDClassifier():
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    number_of_labels : int, how many classes need to classification

    combination_type : int，1 (PRODUCT),0 (MINIMUM)
        T-norm for the Computation of the Compatibility Degree
    rule_weight : int,1 (PCF_IV	，Penalized_Certainty_Factor),0(CF，Certainty_Factor),
                      3(PCF_II，Average_Penalized_Certainty_Factor),3(NO_RW，No_Weights)
    inference_type : 0 ( WINNING_RULE, WINNING_RULEWinning_Rule), 1(ADDITIVE_COMBINATION,Additive_Combination)
          Fuzzy Reasoning Method
    ranges : [[0.0 for y in range (2)] for x in range nVars], nVars=self.__nInputs + Attributes.getOutputNumAttributes(Attributes)


    example :
        Number of Labels = 3
        T-norm for the Computation of the Compatibility Degree = Product
        Rule Weight = Penalized_Certainty_Factor
        Fuzzy Reasoning Method = Winning_Rule
        ranges = [[0.0 for y in range (2)] for x in range 4]
                 [4,3         7.9
                  2.0         4.4
                  1.0         6.9
                  0.1         2.5]


    """

    # algorithm parameters
    # int
    number_of_labels = 0
    population_size = 0
    depth = 0
    k_parameter = 0
    max_trials = 0
    type_inference = 0
    bits_gen = 0

    minsup = 0.0
    minconf = 0.0
    alpha = 0.0
    something_wrong = False

    train_mydataset = None
    val_mydataset = None
    test_mydataset = None

    output_tr = ""
    output_tst = ""

    file_db = ""
    file_rb = ""
    file_time = ""
    file_hora = ""
    data_string = ""
    file_rules = ""
    evolution = ""

    rules_stage1 = 0
    rules_stage2 = 0
    rules_stage3 = 0

    data_base = None
    rule_base = None

    apriori = None

    pop = None
    start_time = 0
    total_time = 0

    # algorithm parameters
    # int
    nlabels = 0
    negative_confident_value = 0
    negative_rule_number = None
    zone_confident = 0
    seed_int =None
    granularity_rule_Base_array=[]


    def __init__(self, prepare_parameter):
        print("__init__ of Fuzzy_Chi begin...")
        self.start_time = datetime.datetime.now()

        self.train_mydataset = MyDataSet()
        self.val_mydataset = MyDataSet()
        self.test_mydataset = MyDataSet()

        try:

            input_training_file = prepare_parameter.get_input_training_files()
            print("Reading the training set: " + input_training_file)

            self.train_mydataset.read_classification_set(input_training_file, True, prepare_parameter.file_path)
            print("Reading the validation set: ")
            input_validation_file = prepare_parameter.get_validation_input_file()
            self.val_mydataset.read_classification_set(input_validation_file, True, prepare_parameter.file_path)
            print("Reading the test set: ")
            self.test_mydataset.read_classification_set(prepare_parameter.get_input_test_files(), False,
                                                        prepare_parameter.file_path)
            print(" ********* test_mydataset.myDataSet read_classification_set finished !!!!!! *********")
        except IOError as ioError:
            print("I/O error: " + str(ioError))
            self.something_wrong = True
        except Exception as e:
            print("Unexpected error:" + str(e))
            self.something_wrong = True

        self.something_wrong = self.something_wrong or self.train_mydataset.has_missing_attributes()
        self.output_tr = prepare_parameter.get_training_output_file()
        self.output_tst = prepare_parameter.get_test_output_file()

        output_file_folder = "results"

        file_db_name = prepare_parameter.get_output_file(0)
        file_rb_name = prepare_parameter.get_output_file(1)

        self.file_db= os.path.join(prepare_parameter.result_path , output_file_folder + "\\"+file_db_name)

        self.file_rb =os.path.join(prepare_parameter.result_path , output_file_folder + "\\"+file_rb_name)

        self.file_db =os.getcwd() + self.file_db
        self.file_rb =os.getcwd() + self.file_rb

        self.data_string = prepare_parameter.get_input_training_files()

        output_file = prepare_parameter.get_output_file(1)
        # print("output_file is : " + output_file)
       
        self.file_time =os.getcwd()+  prepare_parameter.result_path + "\\" + output_file_folder + "\\" +"time.txt"
        self.file_hora =os.getcwd() + prepare_parameter.result_path + "\\" + output_file_folder +"\\" +"hora.txt"
        self.file_rules = os.getcwd()+ prepare_parameter.result_path +"\\"+ output_file_folder + "\\" +"rules.txt"
        # Now we parse the parameters long
        self.seed_int = int(float(prepare_parameter.get_parameter(0)))

        self.nlabels = int(prepare_parameter.get_parameter(1))
        self.minsup = float(prepare_parameter.get_parameter(2))
        self.minconf = float(prepare_parameter.get_parameter(3))
        self.depth = int(prepare_parameter.get_parameter(4))
        self.k_parameter = int(prepare_parameter.get_parameter(5))
        self.max_trials = int(prepare_parameter.get_parameter(6))
        self.population_size = int(prepare_parameter.get_parameter(7))
        if self.population_size % 2 > 0:
            self.population_size = self.population_size + 1
        self.alpha = float(prepare_parameter.get_parameter(8))
        self.bits_gen = int(prepare_parameter.get_parameter(9))
        self.type_inference = int(prepare_parameter.get_parameter(10))
        # javarandom.Random(self.seed_int)
        random.seed(self.seed_int)

    def fit(self, X, y):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        In fit function it will generate the rules and store it
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        if self.something_wrong:  # We do not execute the program
            print("An error was found, the data-set have missing values")
            print("Please remove the examples with missing data or apply a MV preprocessing.")
            print("Aborting the program")
        # We should not use the statement: System.exit(-1);
        else:
            print("No errors, Execute in FarcHD execute :")
            self.data_base = DataBase()
            self.data_base.init_with_three_parameters(self.nlabels, self.train_mydataset)
            self.rule_base = RuleBase()
            self.rule_base.init_with_five_parameters(self.data_base, self.train_mydataset, self.k_parameter,
                                                     self.type_inference)
            self.apriori = Apriori()
            self.apriori.multiple_init(self.rule_base, self.data_base, self.train_mydataset, self.minsup, self.minconf,
                                       self.depth)
            self.apriori.generate_rb()
            self.rules_stage1 = self.apriori.get_rules_stage1()
            self.rules_stage2 = self.rule_base.get_size()
            print("self.rules_stage1")
            print(self.rules_stage1)
            print("self.rules_stage2")
            print(self.rules_stage2)

            self.pop = Populate()

            self.pop.init_with_multiple_parameters(self.seed_int,self.train_mydataset, self.data_base, self.rule_base,
                                                   self.population_size, self.bits_gen, self.max_trials, self.alpha)
            self.pop.generation()

            print("Building classifier")
            self.rule_base = self.pop.get_best_RB()

            self.rules_stage3 = int(self.rule_base.get_size())

            print("Begin the  negative rule generation ")
            self.data_base.save_file(self.file_db)
            self.rule_base.save_file(self.file_rb)

            self.rule_base.generate_negative_rules(self.train_mydataset, self.negative_confident_value,self.zone_confident)

            self.negative_rule_number = len(self.rule_base.negative_rule_base_array)


            print("Begin the  granularity rule generation ")
            print("self.nlabels "+str(self.nlabels))

            granularity_rule = GranularityRule(self.train_mydataset,self.nlabels,
                self.file_db,self.file_rb,self.val_mydataset,
                self.output_tr,self.output_tst,self.rule_base,self.k_parameter,self.data_base,self.test_mydataset,
                self.val_mydataset,self.type_inference,self.minsup,self.minconf,self.depth,self.seed_int,self.population_size,self.bits_gen,self.alpha,self.max_trials)

            self.granularity_rule_Base_array = granularity_rule.get_granularity_rules(self.negative_rule_number)
            


            #

            #  Finally we should fill the training and test  output files
            self.do_output(self.val_mydataset, self.output_tr)
            self.do_output(self.test_mydataset, self.output_tst)
            current_millis = int(round(time.time() * 1000))

            # int(datetime.datetime.utcnow().timestamp())

            self.total_time = current_millis - int(self.start_time.utcnow().timestamp())
            self.write_time()
            self.write_rules()

      
            print("Algorithm Finished")

            # Return the classifier
            return self

    # """
    #    * It generates the output file from a given dataset and stores it in a file
    #    * @param dataset myDataset input dataset
    #    * @param filename String the name of the file
    #    *
    #    * @return The classification accuracy
    # """

    def do_output(self, mydataset, filename):

        output = mydataset.copy_header()  # we insert the header in the output file
        # We write the output for each example
        for i in range(0, mydataset.get_ndata()):
            # for classification:
            output = output + mydataset.get_output_as_string_with_pos(i) + " " + self.classification_output(
                mydataset.get_example(i)) + "\n"

        if os.path.isfile(filename):
            # print("File exist")
            output_file = open(filename, "a+")
        else:
            # print("File not exist")
            output_file = open(filename, "w+")

        output_file.write(output)

    # * It returns the algorithm classification output given an input example
    # * @param example double[] The input example
    # * @return String the output generated by the algorithm

    def classification_output(self, example):
        output = "?"
        # Here we should include the algorithm directives to generate the
        # classification output from the input example

        class_out = self.rule_base.frm(example)

        if class_out >= 0:
            output = self.train_mydataset.get_output_value(class_out)

        return output

    def write_time(self):
        aux = None  # int
        seg = None  # int
        min_value = None  # int
        hor = None  # int

        string_out = "" + str(self.total_time / 1000) + "  " + self.data_string + "\n"
        file = open(self.file_time, "a+")
        file.write(string_out)
        self.total_time /= 1000
        seg = self.total_time % 60
        self.total_time = self.total_time / 60
        min_value = self.total_time % 60
        hor = self.total_time / 60
        string_out = ""
        if hor < 10:
            string_out = string_out + "0" + str(hor) + ":"
        else:
            string_out = string_out + str(hor) + ":"
        if min_value < 10:
            string_out = string_out + "0" + str(min_value) + ":"
        else:
            string_out = string_out + str(min_value) + ":"

        if seg < 10:
            string_out = string_out + "0" + str(seg)
        else:
            string_out = string_out + str(seg)

        string_out = string_out + "  " + self.data_string + "\n"

        file = open(self.file_hora, "a+")
        file.write(string_out)

    """ 
     * Add all the rules generated by the classifier to fileRules file.
     """

    def write_rules(self):

        string_out = "" + str(self.rules_stage1) + " " + str(self.rules_stage2) + " " + str(self.rules_stage3) + "\n"

        file = open(self.file_rules, "a+")
        file.write(string_out)

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen udring fit.
        """

        # Input validation
        X = check_array(X, accept_sparse=True)
        selected_array = [1, 1, 1, 1, 1, 1, 1, 1]
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'], 'is_fitted_')

        row_num = X.shape[0]
        predict_y = np.empty([row_num, 1], dtype=np.int32)

        for i in range(0, row_num):
            predict_y[i] = self.rule_base.frm_ac_with_two_parameters(X[i], selected_array)
        print("predict_y is :")
        print(predict_y)

        return predict_y[i]

    def score(self, test_X, test_y):
        """ A reference implementation of score function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The test input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each test sample is the label of the closest test sample
            seen udring fit.
        """

        # Input validation
        test_X = check_array(test_X, accept_sparse=True)
        selected_array = [1, 1, 1, 1, 1, 1, 1, 1]
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'], 'is_fitted_')

        row_num = test_X.shape[0]
        print("row_num in score is :" + str(row_num))
        predict_y = np.empty([row_num, 1], dtype=np.int32)
        hits = 0

        for i in range(0, row_num):
            predict_y[i] = self.rule_base.frm_ac_with_two_parameters(test_X[i], selected_array)

            print("predict_y[" + str(i) + "] is :" + str(predict_y[i]))
            print("test_y[" + str(i) + "] is :" + str(test_y[i]))

            if predict_y[i] == test_y[i]:
                hits = hits + 1

        print("predict_y normal rules in score is :")
        score = 1.0 * hits / row_num
        print(score)

        return score
