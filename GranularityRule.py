from Apriori import Apriori
from DataBase import DataBase
from ExampleWeight import ExampleWeight
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
    WINNING_RULE = 0
    granularity_confident_value = 0
    PRODUCT = 1 
    ruleWeight = 1

    more_granularity = True
    negative_rule_number = 0

    k_parameter = 0
    data_base =  None
    type_inference=  None
    minsup = 0.0
    minconf = 0.0
    depth = 0


    def __init__(self,train_mydataset,nlabels,file_db,file_rb,
                 val_mydataset,outputTr,outputTst,ruleBase,
                 k_parameter,data_base,test_myDataSet,val_myDataSet,type_inference, minsup,minconf,depth):

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
        self.minsup =minsup
        self.minconf =minconf
        self.depth =depth





    
    def prepare(self,negative_rule_number):
          # 1. get the sub train myDataSet from negative rules
            self.granularity_database_array = [DataBase() for x in range(negative_rule_number)]
            self.granularity_rule_Base_array = [RuleBase() for x in range(negative_rule_number)]

            # from negative rule get small_disjunct_train array
            self.extract_small_disjunct_train_array_step_one(self.train_myDataSet)

            # need to get below 4 values:
            # self.train_myDataSet.getnInputs(), self.nLabels,
            # self.train_myDataSet.getRanges(), self.train_myDataSet.getNames()
            nVars = self.train_myDataSet.get_nvars()
            nInputs = self.train_myDataSet.get_ninputs()
            nominal_array = self.train_myDataSet.nominal_array

            integer_array =self.train_myDataSet.integer_array

            while self.more_granularity and self.negative_rule_number > 0:

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


                    #  added by rui for granularity rules
                    self.granularity_rule_Base_array[i] = RuleBase()
                    self.granularity_rule_Base_array[i].init_with_five_parameters(self.data_base,self.train_myDataSet,self.k_parameter,self.inferenceType)

                    self.granularity_database_array[i].init_with_three_parameters(self.nLabels, self.train_myDataSet)


                self.generate_granularity_rules()
                self.prunerules_granularity_rules()

                print("self.fileDB = " + str(self.fileDB))
                print("self.fileRB = " + str(self.fileRB))
                for i in range(0, self.negative_rule_number):
                    self.granularity_database_array[i].save_file(self.fileDB)

                # Finally we should fill the training and test output files with granularity rule result

                accTra = self.doOutput(self.val_myDataSet, self.outputTr, True)
                accTst = self.doOutput(self.test_myDataSet, self.outputTst,True)
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
    def doOutput(self, dataset, filename, granularity_rule):
        try:
            output = ""
            hits = 0
            self.output = dataset.copyHeader()  # we insert the header in the output file
            # We write the output for each example
            # print("before loop in Fuzzy_Chi")
            data_number = dataset.getnData()
            print("in doOutput dataset.getnData()" + str(dataset.getnData()))
            for i in range(0, data_number):
                # print(" In the doOutput the loop number i is  " + str(i))
                # for classification:
                # print("before classificationOutput in Fuzzy_Chi")
                class_out_here = None
                if granularity_rule:
                    for j in range(0, self.negative_rule_number):

                        # classOut = self.classification_Output_granularity(dataset.getExample(j), j)
                        classOut = self.classification_Output_pruned_granularity(dataset.getExample(i), j)
                        if classOut is not "?":
                            class_out_here = classOut

                    if class_out_here is None:  #
                        classOut = self.classificationOutput(dataset.getExample(i))
                    else:
                        classOut = class_out_here

                else:
                    classOut = self.classificationOutput(dataset.getExample(i))

                # arrived here, print("before getOutputAsStringWithPos in Fuzzy_Chi")
                self.output = self.output + dataset.getOutputAsStringWithPos(i) + " " + classOut + "\n"
                # not arrive here, print("before getOutputAsStringWithPos in Fuzzy_Chi")
                if dataset.getOutputAsStringWithPos(i) == classOut:
                    hits = hits + 1
            # print("before open file in Fuzzy_Chi")
            file = open(filename, "w+")
            file.write(output)
            file.close()
        except Exception as excep:
            print("There is exception in doOutput in Fuzzy chi class !!! The exception is :" + str(excep))
        if dataset.size() != 0:
            print(" in doOutput the hits is "+str(hits)+" ,the dataset.size() is "+str(dataset.size()))
            return 1.0 * hits / dataset.size()
        else:
            return 0

    
    def classificationOutput(self, example):
        self.output = "?"
        # Here we should include the algorithm directives to generate the
        # classification output from the input example
        classOut = self.ruleBase.FRM(example)
        if classOut >= 0:
            # print("In Fuzzy_Chi,classOut >= 0, to call getOutputValue")
            self.output = self.train_myDataSet.getOutputValue(classOut)
        return self.output


    def classification_Output_granularity(self, example, zone_area_number):
        self.output = "?"
        # Here we should include the algorithm directives to generate the
        # classification output from the input example

        classOut = self.granularity_rule_Base_array[zone_area_number].FRM_Granularity(example)
        if classOut >= 0:
            # print("In Fuzzy_Chi,classOut >= 0, to call getOutputValue")
            self.output = self.my_dataset_train_sub_zone[zone_area_number].getOutputValue(classOut)
        return self.output

    
    def classification_Output_pruned_granularity(self, example, zone_area_number):
        self.output = "?"
        # Here we should include the algorithm directives to generate the
        # classification output from the input example

        classOut = self.granularity_rule_Base_array[zone_area_number].FRM_Pruned_Granularity(example)
        if classOut >= 0:
            # print("In Fuzzy_Chi,classOut >= 0, to call getOutputValue")
            self.output = self.my_dataset_train_sub_zone[zone_area_number].getOutputValue(classOut)
        return self.output

    def generate_granularity_rules(self):
        # from negative rule get small_disjunct_train array
        # for each small disjunct train generate positive rules, save into priority rule base
        for i in range(0, self.negative_rule_number):
            sub_train_zone = self.my_dataset_train_sub_zone[i]
            self.generation_rule_step_two(sub_train_zone, sub_train_zone.size(), i)
        for i in range(0, self.negative_rule_number):
            print(" The loop i number is :" + str(i))
            self.granularity_rule_Base_array[i].write_File_for_granularity_rule(self.fileRB)

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
            #print(" output_integer length is： " + str(len(output_integer[i])))
            #print(" x_array length is： " + str(len(x_array[i])))

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
            #print("nclasses_number of " + str(k) + " is  :" + str(nclasses_number))
            self.my_dataset_train_sub_zone[k].set_nclasses(nclasses_number)
            number_of_data = self.my_dataset_train_sub_zone[k].size()
            #print(" The my_dataset_train_sub_zone " + str(k) + " number is :" + str(number_of_data))

    def generation_rule_step_two(self, sub_train, sub_zone_number, area_number):
        print("In generation, the area_number is :" + str(area_number))
        print("In generation, the size of sub train is :" + str(sub_train.size()))

        self.granularity_rule_base = RuleBase()
        self.granularity_rule_base.init_with_five_parameters(self.data_base, sub_train, self.k_parameter,
                                                 self.type_inference)
        self.apriori = Apriori()
        self.apriori.multiple_init(self.granularity_rule_base, self.data_base, sub_train, self.minsup, self.minconf,
                                   self.depth)
        self.apriori.generate_rb()
        self.granularity_rules_stage1 = self.apriori.get_rules_stage1()
        self.granularity_rules_stage2 = self.granularity_rule_base.get_size()

        """  
        example_weight: ExampleWeight = []
        for i in range(0, sub_train.size()):
            example_weight.append(ExampleWeight(self.k_parameter))
    
      
        for i in range(0, sub_train.size()):
            granularity_rule = self.granularity_rule_Base_array[area_number].search_for_best_antecedent(
                sub_train.get_example(i), sub_train.get_output_as_integer_with_pos(i),self.nLabels)
            self.granularity_data_row_array.append(granularity_rule.data_row_here)
            granularity_rule.calculate_wracc(sub_train, example_weight)
            if not (self.granularity_rule_Base_array[area_number].duplicated_granularity_rule(granularity_rule)) and (
                    granularity_rule.wracc > self.granularity_confident_value):
                granularity_rule.granularity_sub_zone = sub_zone_number
                self.granularity_rule_Base_array[area_number].granularity_rule_Base.append(granularity_rule)      
        """
        print("The total granularity_data_row_array is " + str(len(self.granularity_data_row_array)))
        print(" In area_number " + str(area_number) + " ,The total granularity_rule_Base rule number is  : " + str(
            len(self.granularity_rule_Base_array[area_number].granularity_rule_Base)))

    def prunerules_granularity_rules(self):
        for i in range(0, self.negative_rule_number):
            print("in prunerules_granularity_rules the i is: " + str(i))
            negative_rule = self.ruleBase.negative_rule_base_array[i]
            for j in range(0, len(self.granularity_rule_Base_array[i].granularity_rule_Base)):
                granularity_rule = self.granularity_rule_Base_array[i].granularity_rule_Base[j]

                if granularity_rule.weight > 0:
                    #if negative_rule.class_value == granularity_rule.class_value:
                    self.granularity_rule_Base_array[i].granularity_prune_rule_base.append(granularity_rule)
                    print(" Added a new pruned granularity rule in  granularity_prune_rule_base, rule weight is :" + str(granularity_rule.weight))
        for i in range(0, self.negative_rule_number):
            print(" The loop i number is :" + str(i))
            self.granularity_rule_Base_array[i].write_File_for_pruned_granularity_rule(self.fileRB)

    
    def decide_more_granularity_or_not(self):
        # print("compare granularity rule and negative rule to see if negative granularity rule has been generated ")
        for i in range(0, self.negative_rule_number):
            negative_rule = self.ruleBase.negative_rule_base_array[i]
            for j in range(0, len(self.granularity_rule_Base_array[i].granularity_rule_Base)):
                granularity_rule = self.granularity_rule_Base_array[i].granularity_rule_Base[j]
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
