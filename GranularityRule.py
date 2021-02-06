from Apriori import Apriori
from DataBase import DataBase
from ExampleWeight import ExampleWeight
from FarcHDSteps import FarcHDSteps
from RuleBase import RuleBase
from MyDataSet import MyDataSet


class GranularityRule:
    my_dataset_train_sub_zone = []
    granularity_database_array = []
    train_myDataSet = None

    nLabels = None
    inferenceType = None
    fileDB = None
    fileRB = None
    test_myDataSet = None
    val_myDataSet = None
    outputTr = None
    outputTst = None
    ruleBase = None
    nClasses = None
    granularity_data_row_array = None
    granularity_data_row_array = []
    granularity_rule_base = None
    granularity_rule_Base_array = []
    WINNING_RULE = 0
    granularity_confident_value = 0
    PRODUCT = 1
    ruleWeight = 1

    more_granularity = True
    negative_rule_number = 0

    k_parameter = 0
    data_base = None
    type_inference = None
    minsup = 0.0
    minconf = 0.0
    depth = 0

    seed_int = 0
    population_size = 0
    bits_gen = 0
    alpha = 0
    max_trials = 0
    max_granularity_degree = None
    normal_rule_degree = None

    def __init__(self, train_mydataset, nlabels, file_db, file_rb,
                 val_mydataset, outputTr, outputTst, ruleBase,
                 k_parameter, data_base, test_myDataSet, val_myDataSet,
                 type_inference, minsup, minconf, depth, seed_int, population_size, bits_gen, alpha, max_trials):

        self.train_myDataSet = train_mydataset
        self.nClasses = self.train_myDataSet.get_nclasses()

        self.inferenceType = self.WINNING_RULE
        self.combinationType = self.PRODUCT

        self.val_myDataSet = val_mydataset
        self.outputTr = outputTr
        self.outputTst = outputTst
        self.ruleBase = ruleBase

        self.fileDB = file_db
        self.fileRB = file_rb
        # Now we parse the parameters

        # self.nLabels = parameters.getParameter(0)
        self.nLabels = nlabels
        self.k_parameter = k_parameter
        self.data_base = data_base

        self.test_myDataSet = test_myDataSet
        self.val_myDataSet = val_myDataSet
        self.type_inference = type_inference
        self.minsup = minsup
        self.minconf = minconf
        self.depth = depth

        self.seed_int = seed_int
        self.population_size = population_size
        self.bits_gen = bits_gen
        self.alpha = alpha
        self.max_trials = max_trials

    def get_granularity_rules(self, negative_rule_number):
        # 1. get the sub train myDataSet from negative rules
        self.granularity_database_array = [DataBase() for x in range(negative_rule_number)]
        self.granularity_rule_Base_array = []

        # from negative rule get small_disjunct_train array
        self.extract_small_disjunct_train_array_step_one(self.train_myDataSet)

        # need to get below 4 values:
        # self.train_myDataSet.getnInputs(), self.nLabels,
        # self.train_myDataSet.getRanges(), self.train_myDataSet.getNames()
        nVars = self.train_myDataSet.get_nvars()
        nInputs = self.train_myDataSet.get_ninputs()
        nominal_array = self.train_myDataSet.nominal_array

        integer_array = self.train_myDataSet.integer_array

        # while self.more_granularity and self.negative_rule_number > 0:

        for i in range(0, self.negative_rule_number):
            # 2. for each sub train myDataSet, do self.granularity_data_base[i]= DataBase()
            self.granularity_database_array[i] = DataBase()
            # 3. self.granularity_data_base[i].setMultipleParameters(......)

            self.my_dataset_train_sub_zone[i].set_nvars(nVars)
            self.my_dataset_train_sub_zone[i].set_ninputs(nInputs)
            self.my_dataset_train_sub_zone[i].nominal_array = nominal_array
            self.my_dataset_train_sub_zone[i].integer_array = integer_array

            print("size of sub zone :" + str(self.my_dataset_train_sub_zone[i].size()))
            sub_x_array = self.my_dataset_train_sub_zone[i].get_x()

            self.granularity_database_array[i].init_with_three_parameters(self.nLabels, self.my_dataset_train_sub_zone[i])

        self.generate_granularity_rules()
        self.prunerules_granularity_rules()

        print("self.fileDB = " + str(self.fileDB))
        print("self.fileRB = " + str(self.fileRB))
        for i in range(0, self.negative_rule_number):
            self.granularity_database_array[i].save_file(self.fileDB)

        # Finally we should fill the training and test output files with granularity rule result

        accTra = self.doOutput(self.val_myDataSet, self.outputTr)
        accTst = self.doOutput(self.test_myDataSet, self.outputTst)
        print("Accuracy for granularity  rules obtained in training data is : " + str(accTra))
        print("Accuracy for granularity  rules obtained in test data is: " + str(accTst))
        # print("Accuracy for normal rules obtained in test: " + str(accTst))
        # self.nLabels = int(self.nLabels) + 1
        # print(" self.nLabels after being added by one " + str(self.nLabels))
        self.decide_more_granularity_or_not()

    # """
    #    * It generates the output file from a given dataset and stores it in a file
    #    * @param dataset myDataset input dataset
    #    * @param filename String the name of the file
    #    *
    #    * @return The classification accuracy
    # """
    def doOutput(self, dataset, filename):
        final_class_out =None
        granularity_rule = True
        try:
            output = ""
            hits = 0
            self.output = dataset.copyHeader()  # we insert the header in the output file
            # We write the output for each example
            # print("before loop in Fuzzy_Chi")
            data_number = dataset.get_ndata()
            print("in doOutput dataset.getnData()" + str(dataset.get_ndata()))
            for i in range(0, data_number):
                # print(" In the doOutput the loop number i is  " + str(i))
                # for classification:
                # print("before classificationOutput in Fuzzy_Chi")
                class_out_here = None
                if granularity_rule:
                    max_degree=0

                    for j in range(0, self.negative_rule_number):
                        print("before classification_Output_granularity")

                        classOut = self.classification_Output_granularity(dataset.get_example(i), j)
                        degree_new = self.max_granularity_degree

                        print("after classification_Output_granularity")
                        # classOut = self.classification_Output_pruned_granularity(dataset.getExample(i), j)
                        if classOut is not "?":
                            if degree_new > max_degree:
                                class_out_here = classOut

                    #if class_out_here is None:  #
                    print("if before class out here is None,  classificationOutput")
                    classOut_normal = self.classificationOutput(dataset.get_example(i))
                    print("if after class out here is None,  classificationOutput")
                    #else:
                    if degree_new > self.normal_rule_degree:
                        final_class_out = class_out_here
                    else:
                        final_class_out = classOut_normal


                    print(" classOut  in doOutput is :" + str(final_class_out))
                    if final_class_out >= 0:
                        print(" final_class_out :" + str(final_class_out))
                        output_with_name = self.train_myDataSet.get_output_value(final_class_out)

                    print(" granularity_rule output_with_name is :" + str(output_with_name))




                print("before get_output_as_string_with_pos in doOutput")
                self.output = str(self.output) + dataset.get_output_as_string_with_pos(i) + " " + str(final_class_out)  + "\n"
                print("after get_output_as_string_with_pos in doOutput")
                dataset_output =dataset.get_output_as_string_with_pos(i)
                print("dataset_output is :"+str(dataset_output))
                print("output_with_name is :" + str(output_with_name))
                if dataset_output == output_with_name:
                    hits = hits + 1
            # print("before open file in Fuzzy_Chi")
            file = open(filename, "w+")
            file.write(output)
            file.close()
        except Exception as excep:
            print("There is exception in doOutput in Granularity rule class !!! The exception is :" + str(excep))
        if dataset.size() != 0:
            print(" in doOutput the hits is " + str(hits) + " ,the dataset.size() is " + str(dataset.size()))
            return 1.0 * hits / dataset.size()
        else:
            return 0

    def classificationOutput(self, example):
        self.output = "?"
        # Here we should include the algorithm directives to generate the
        # classification output from the input example
        selected_array = [1, 1, 1, 1, 1, 1, 1, 1]
        classOut = self.ruleBase.frm_ac_with_two_parameters(example,selected_array)
        self.normal_rule_degree = self.ruleBase.frm_ac_max_degree_value

        print("classOut in classificationOutput is "+str(classOut))
        if classOut >= 0:
            # print("In Fuzzy_Chi,classOut >= 0, to call getOutputValue")
            self.output = self.train_mydataset.get_output_as_string_with_pos(classOut)
        return self.output

    def classification_Output_granularity(self, example, zone_area_number):
        self.output = "?"

        # Here we should include the algorithm directives to generate the
        # classification output from the input example
        print("before FRM_Granularity")
        selected_array = [1, 1, 1, 1, 1, 1, 1, 1]
        classOut = self.granularity_rule_Base_array[zone_area_number].frm_ac_with_two_parameters(example,selected_array)

        self.max_granularity_degree = self.granularity_rule_Base_array[zone_area_number].frm_ac_max_degree_value
        print("in classification_Output_granularity  max_granularity_degree is "+str(self.max_granularity_degree))

        return classOut

    def classification_Output_pruned_granularity(self, example, zone_area_number):
        self.output = "?"
        # Here we should include the algorithm directives to generate the
        # classification output from the input example

        classOut = self.granularity_rule_Base_array[zone_area_number].rule_base_array.FRM(example)
        if classOut >= 0:
            # print("In Fuzzy_Chi,classOut >= 0, to call getOutputValue")
            self.output = self.my_dataset_train_sub_zone[zone_area_number].get_output_as_string_with_pos(classOut)
        return self.output

    def generate_granularity_rules(self):
        # from negative rule get small_disjunct_train array
        # for each small disjunct train generate positive rules, save into priority rule base
        for i in range(0, self.negative_rule_number):
            sub_train_zone = self.my_dataset_train_sub_zone[i]
            sub_train_zone.compute_instances_per_class()

            self.generation_rule_step_two(sub_train_zone, sub_train_zone.size(), i)
        granularity_rule_base_number = len(self.granularity_rule_Base_array)
        for i in range(0, granularity_rule_base_number):
            print(" The loop i number is :" + str(i))
            self.granularity_rule_base = self.granularity_rule_Base_array[i].rule_base_array
            self.granularity_rule_Base_array[i].write_File_for_granularity_rule(self.fileRB, self.granularity_rule_base)

    def extract_small_disjunct_train_array_step_one(self, train):
        self.negative_rule_number = len(self.ruleBase.negative_rule_base_array)
        x_array = [[] for x in range(self.negative_rule_number)]
        output_integer = [[] for x in range(train.size())]
        output = [[] for x in range(train.size())]
        train_x_array = train.get_x()
        # print("generate granularity rules begin :")
        data_row_number = len(self.ruleBase.data_row_array)

        for m in range(0, self.negative_rule_number):
            x_array[m] = []
            output_integer[m] = []
            output[m] = []
        self.my_dataset_train_sub_zone = [MyDataSet() for x in range(self.negative_rule_number)]

        for i in range(0, self.negative_rule_number):
            for dr in range(0, data_row_number):
                same_label_num = 0
                negative_rule_here = self.ruleBase.negative_rule_base_array[i]
                number_label_of_rule = 0
                for j in range(0, self.ruleBase.n_variables):
                    # print(" self.data_row_array[dr].label_values[j]" + str(self.ruleBase.data_row_array[dr].label_values[j]))
                    # print(" negative_rule_here.antecedent[j]: " + str(negative_rule_here.antecedent[j]))
                    antecedent_value = negative_rule_here.antecedent[j]
                    label_value = self.ruleBase.data_row_array[dr].label_values[j]
                    if not antecedent_value == -1:
                        number_label_of_rule = number_label_of_rule + 1
                        if label_value == antecedent_value:
                            same_label_num = same_label_num + 1
                if same_label_num == number_label_of_rule:
                    # added the related __X in new sub zone train data set
                    # print("train_x_array[dr]" + str(train_x_array[dr]))
                    x_array[i].append(train_x_array[dr])
                    output_integer[i].append(train.get_output_as_integer_with_pos(dr))
                    output[i].append(train.get_output_as_integer_with_pos(dr))
            # print(" output_integer length is： " + str(len(output_integer[i])))
            # print(" x_array length is： " + str(len(x_array[i])))

        for k in range(0, self.negative_rule_number):
            print(" my_dataset_train_sub_zone[ " + str(k) + " ] :" + str(self.my_dataset_train_sub_zone[k]))
            num_sub_zone = len(x_array[k])
            # set my data set X array
            self.my_dataset_train_sub_zone[k].set_x(x_array[k])
            self.my_dataset_train_sub_zone[k].set_output_integer_array(output_integer[k])
            self.my_dataset_train_sub_zone[k].set_output_array(output[k])
            self.my_dataset_train_sub_zone[k].set_ndata(num_sub_zone)
            print("num_sub_zone " + str(k) + " is  :" + str(num_sub_zone))
            # set the rule base nClasses value
            # nclasses_number = self.my_dataset_train_sub_zone[k].calculate_nclasses_for_small_granularity_zone(output_integer[k])
            nclasses_number = self.nClasses
            # print("nclasses_number of " + str(k) + " is  :" + str(nclasses_number))
            self.my_dataset_train_sub_zone[k].set_nclasses(nclasses_number)
            number_of_data = self.my_dataset_train_sub_zone[k].size()
            # print(" The my_dataset_train_sub_zone " + str(k) + " number is :" + str(number_of_data))

    def generation_rule_step_two(self, sub_train, sub_zone_number, area_number):

        print("In generation, the area_number is :" + str(area_number))
        print("In generation, the size of sub train is :" + str(sub_train.size()))

        farchd = FarcHDSteps(self.nLabels, sub_train, self.k_parameter, self.type_inference, self.minsup, self.minconf,
                             self.depth, self.alpha, self.max_trials)
        farchd.execute_FarcHd(self.seed_int, self.population_size, self.bits_gen)
        rule_number = farchd.rule_base.get_size()
        print("In generation, in sub_zone_number :" + str(sub_zone_number) + " ,granalarity rules number is :" + str(
            rule_number))

        if rule_number > 0:
            #  added by rui for granularity rules
            #  rule_base = RuleBase()
            #  rule_base.init_with_five_parameters(self.data_base, self.train_myDataSet,

            self.granularity_rule_Base_array.append(farchd.rule_base)



    def prunerules_granularity_rules(self):
        for i in range(0, self.negative_rule_number):
            print("in prunerules_granularity_rules the i is: " + str(i))
            negative_rule = self.ruleBase.negative_rule_base_array[i]
            for j in range(0, len(self.granularity_rule_Base_array[i].rule_base_array)):
                granularity_rule = self.granularity_rule_Base_array[i].rule_base_array[j]

                if granularity_rule.wracc > 0.2:
                    if negative_rule.class_value == granularity_rule.class_value:
                        self.granularity_rule_Base_array[i].granularity_prune_rule_base.append(granularity_rule)
                        print(" Added a new pruned granularity rule in  granularity_prune_rule_base, rule weight is :" + str(granularity_rule.weight))
        granularity_rule_base_number = len(self.granularity_rule_Base_array)
        for i in range(0, granularity_rule_base_number):
            print(" The loop i number is :" + str(i))
            rule_base =self.granularity_rule_Base_array[i]
            rule_base.write_File_for_pruned_granularity_rule(self.fileRB,rule_base.granularity_prune_rule_base)

    def decide_more_granularity_or_not(self):
        # print("compare granularity rule and negative rule to see if negative granularity rule has been generated ")
        for i in range(0, self.negative_rule_number):
            negative_rule = self.ruleBase.negative_rule_base_array[i]
            for j in range(0, len(self.granularity_rule_Base_array[i].rule_base_array)):
                granularity_rule = self.granularity_rule_Base_array[i].rule_base_array[j]
                # print(" negative_rule.class_value" + str(negative_rule.class_value))
                # print(" granularity_rule.class_value" + str(granularity_rule.class_value))
                if negative_rule.class_value == granularity_rule.class_value:
                    self.more_granularity = False
                    # print(" Set the self.more_granularity = False")
                    break
                if not self.more_granularity:
                    break
            if not self.more_granularity:
                break
