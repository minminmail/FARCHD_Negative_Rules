from decimal import Decimal


from DataBase import DataBase
from Rule import Rule
from Logger import Logger
from ExampleWeight import ExampleWeight
import gc
from DataRow import DataRow


# * This class contains the representation of a Rule Set
# *
# * @version 1.0
# * @since JDK1.5

class RuleBase:
    rule_base_array = []
    train_myDataSet = None
    # added by rui for negative rule
    granularity_rule_Base = []
    granularity_prune_rule_base = []
    negative_rule_base_array = []
    data_base = DataBase()
    n_variables = None

    n_labels = None
    ruleWeight = None
    inferenceType = None

    compatibilityType = None
    names = []
    classes = []
    data_row_array = []

    fitness = None
    k_value = None
    default_rule = None
    nuncover = None
    nuncover_class_array = []
    logger = None
    frm_ac_max_degree_value = None

    # /**
    #  * Rule Base Constructor
    #  * @param dataBase DataBase the Data Base containing the fuzzy partitions
    #  * @param inferenceType int the inference type for the FRM
    #  * @param compatibilityType int the compatibility type for the t-norm
    #  * @param ruleWeight int the rule weight heuristic
    #  * @param names String[] the names for the features of the problem
    #  * @param classes String[] the labels for the class attributes
    #  */
    def __init__(self):
        self.logger = Logger.set_logger( )
        pass

    def init_with_five_parameters(self, data_base_pass, train_myDataset_pass, K_int, inferenceType_pass):
        self.rule_base_array = []
        self.data_base = data_base_pass
        self.train_myDataSet = train_myDataset_pass
        self.n_variables = self.data_base.num_variables()
        self.fitness = 0.0
        self.k_value = K_int
        self.inferenceType = inferenceType_pass
        self.default_rule = -1
        self.nuncover = 0
        self.nuncover_class_array = [0 for x in range(self.train_myDataSet.get_nclasses())]

    # * It checks if a specific rule is already in the rule base
    # * @param r Rule the rule for comparison
    # * @return boolean true if the rule is already in the rule base, false in other case

    def duplicated(self, rule):
        i = 0
        found = False
        while (i < len(self.rule_base_array)) and (not found):
            found = self.rule_base_array[i].comparison(rule)
            i = i + 1
        return found

    def duplicated_granularity_rule(self, rule):
        i = 0
        found = False
        while (i < len(self.granularity_rule_Base)) and (not found):
            found = self.granularity_rule_Base[i].comparison(rule)
            i = i + 1
        return found

    def duplicated_negative_rule(self, rule):
        i = 0
        found = False
        while (i < len(self.negative_rule_base_array)) and (not found):
            found = self.negative_rule_base_array[i].comparison(rule)
            i = i + 1
        return found

    # * It prints the rule base into an string
    # * @return String an string containing the rule base

    def printString(self):
        i = 0
        j = 0
        ant = 0
        self.names = self.train_myDataSet.get_names()
        self.classes = self.train_myDataSet.get_classes()
        cadena_string = ""
        cadena_string += "@Number of rules: " + str(len(self.rule_base_array)) + "\n\n"

        for i in range(0, len(self.rule_base_array)):
            rule = self.rule_base_array[i]
            cadena_string += str(i + 1) + ": "

            for j in range(0, self.n_variables):
                if rule.antecedent[j] < 0:
                    pass
                else:
                    break

            if j < self.n_variables and rule.antecedent[j] >= 0:
                cadena_string += self.names[j] + " IS " + rule.data_base.print_here(j, rule.antecedent[j])
                ant = ant + 1
            	
            # print("after if , j is :" + str(j))
            j=j+1
            k =j
            for j in range(k , self.n_variables):

                if rule.antecedent[j] >= 0:
                    cadena_string += " AND " + self.names[j] + " IS " + rule.data_base.print_here(j, rule.antecedent[j])
                    ant = ant + 1

            cadena_string += ": " + self.classes[rule.class_value]
            cadena_string += " CF: " + str(rule.get_confidence()) + "\n"

        cadena_string += "\n\n"

        cadena_string += "@supp and CF:\n\n"
        for i in range(0, len(self.rule_base_array)):
            rule = self.rule_base_array[i]
            cadena_string += str(i + 1) + ": "
            cadena_string += "supp: " + str(rule.get_support()) + " AND CF: " + str(rule.get_confidence()) + "\n"

        # added negative rule print into file
        cadena_string += "@Number of negative rules: " + str(len(self.negative_rule_base_array)) + "\n\n"

        for i in range(0, len(self.negative_rule_base_array)):
            rule = self.negative_rule_base_array[i]
            cadena_string += str(i + 1) + ": "

            for j in range(0, self.n_variables):
                if rule.antecedent[j] < 0:
                    pass
                else:
                    break

            if j < self.n_variables and rule.antecedent[j] >= 0:
                cadena_string += self.names[j] + " IS " + rule.data_base.print_here(j, rule.antecedent[j])
                ant = ant + 1

            # print("after if , j is :" + str(j))
            j = j + 1
            k = j
            for j in range(k, self.n_variables):

                if rule.antecedent[j] >= 0:
                    cadena_string += " AND " + self.names[j] + " IS " + rule.data_base.print_here(j, rule.antecedent[j])
                    ant = ant + 1

            cadena_string += ": " + self.classes[rule.class_value]
            cadena_string += " CF: " + str(rule.get_confidence()) + "\n"

        cadena_string += "\n\n"

        cadena_string += "@supp and CF:\n\n"
        for i in range(0, len(self.negative_rule_base_array)):
            rule = self.negative_rule_base_array[i]
            cadena_string += str(i + 1) + ": "
            cadena_string += "supp: " + str(rule.get_support()) + " AND CF: " + str(rule.get_confidence()) + "\n"

        print("Begin to print rules :" + "\n\n" + cadena_string)

        cadena_string = cadena_string + str(ant * 1.0 / len(self.rule_base_array)) + "\n\n"

        return cadena_string

    def print_granularity_rule_string(self,rule_base):
        cadena_string = ""

        if rule_base is None or len(rule_base) is 0:
            return cadena_string
        else:
            self.granularity_rule_Base = rule_base

            i = 0
            j = 0
            ant = 0
            self.names = self.train_myDataSet.get_names()
            self.classes = self.train_myDataSet.get_classes()


            # added negative rule print into file
            cadena_string += "@Granularity rules: " + str(len(self.granularity_rule_Base)) + "\n\n"

            for i in range(0, len(self.granularity_rule_Base)):
                rule = self.granularity_rule_Base[i]
                cadena_string += str(i + 1) + ": "

                for j in range(0, self.n_variables):
                    if rule.antecedent[j] < 0:
                        pass
                    else:
                        break

                if j < self.n_variables and rule.antecedent[j] >= 0:
                    cadena_string += self.names[j] + " IS " + rule.data_base.print_here(j, rule.antecedent[j])
                    ant = ant + 1

                # print("after if , j is :" + str(j))
                j = j + 1
                k = j
                for j in range(k, self.n_variables):

                    if rule.antecedent[j] >= 0:
                        cadena_string += " AND " + self.names[j] + " IS " + rule.data_base.print_here(j, rule.antecedent[j])
                        ant = ant + 1

                cadena_string += ": " + self.classes[rule.class_value]
                cadena_string += " CF: " + str(rule.get_confidence()) + "\n"

            cadena_string += "\n\n"

            cadena_string += "@supp and CF:\n\n"
            for i in range(0, len(self.granularity_rule_Base)):
                rule = self.granularity_rule_Base[i]
                cadena_string += str(i + 1) + ": "
                cadena_string += "supp: " + str(rule.get_support()) + " AND CF: " + str(rule.get_confidence()) + "\n"

            print("granularity rules rule_base_array cadena_string is:" + cadena_string)
        return cadena_string

    def print_pruned_granularity_rule_string(self,rule_base):
        # added for granularity rules
        cadena_string = ""
        if rule_base is None or len(rule_base) is 0:
            return cadena_string
        else:
            i = 0
            j = 0
            ant = 0
            self.names = self.train_myDataSet.get_names()
            self.classes = self.train_myDataSet.get_classes()


            # added negative rule print into file
            cadena_string += "@Pruned Granularity rules: " + str(len(rule_base)) + "\n\n"

            for i in range(0, len(rule_base)):
                rule = rule_base[i]
                cadena_string += str(i + 1) + ": "

                for j in range(0, self.n_variables):
                    if rule.antecedent[j] < 0:
                        pass
                    else:
                        break

                if j < self.n_variables and rule.antecedent[j] >= 0:
                    cadena_string += self.names[j] + " IS " + rule.data_base.print_here(j, rule.antecedent[j])
                    ant = ant + 1

                # print("after if , j is :" + str(j))
                j = j + 1
                k = j
                for j in range(k, self.n_variables):

                    if rule.antecedent[j] >= 0:
                        cadena_string += " AND " + self.names[j] + " IS " + rule.data_base.print_here(j, rule.antecedent[j])
                        ant = ant + 1

                cadena_string += ": " + self.classes[rule.class_value]
                cadena_string += " CF: " + str(rule.get_confidence()) + "\n"

            cadena_string += "\n\n"

            cadena_string += "@supp and CF:\n\n"
            for i in range(0, len(rule_base)):
                rule = rule_base[i]
                cadena_string += str(i + 1) + ": "
                cadena_string += "supp: " + str(rule.get_support()) + " AND CF: " + str(rule.get_confidence()) + "\n"

            print("pruned granularity rules rule_base_array cadena_string is:" + cadena_string)
            return cadena_string

    # * It writes the rule base into an ouput file
    # * @param filename String the name of the output file

    def writeFile(self, filename):
        print("rule string to save is: " + self.printString())
        outputString = self.printString()
        file = open(filename, "w+")
        file.write(outputString)
        file.close()

    def write_File_for_granularity_rule(self, filename,rule_base):
        with open(filename, 'a') as file_append:
            outputString = "\n" + "\n" + self.print_granularity_rule_string(rule_base)
            file_append.write(outputString)
            file_append.close()

    def write_File_for_pruned_granularity_rule(self, filename,rule_base):
        with open(filename, 'a') as file_append:
            outputString = "\n" + "\n" + self.print_pruned_granularity_rule_string(rule_base)
            file_append.write(outputString)
            file_append.close()

    """
     * Predicts the class value for a given example, using the rules and type of inference stored on the RuleBase. 
     * @param example Example to be predicted.
    """

    def FRM(self, example):

        if self.inferenceType == 0:
            # print("run FRM_WR !")
            return self.FRM_WR(example)
        else:
            # print("run FRM_AC !")
            return self.FRM_AC(example)

    # * Fuzzy Reasoning Method
    # * @param example double[] the input example
    # * @return int the predicted class label (id)

    def frm_two_parameters(self, example,selected_array_pass):
        # print("run frm_two_parameters !")
        if self.inferenceType == 0:

            return self.frm_wr_with_two_parameters(example,selected_array_pass)
        else:
            result = self.frm_ac_with_two_parameters(example,selected_array_pass)
            if result is None:
                print("The results is none ! from frm_ac_with_two_parameters ")
            return result

    # * Winning Rule FRM
    # * @param example double[] the input example
    # * @return int the class label for the rule with highest membership degree to the example

    def FRM_WR(self, example):
        class_value = 0
        max_value = 0.0
        degree = 0.0
        class_value = self.defaultRule

        for i in range(0, len(self.rule_base_array)):
            rule = self.rule_base_array[i]
            degree = rule.matching(example)

            if degree > max_value:
                max_value = degree
                class_value = rule.get_class()

        return class_value

    def frm_wr_with_two_parameters(self, example, selected_array_pass):
        print("run frm_wr_with_two_parameters !")
        class_value = self.default_rule
        max_value = 0.0

        for i in range(0, len(self.rule_base_array)):
            if selected_array_pass[i] > 0:
                rule = self.rule_base_array[i]
                degree = rule.matching(example)
                if degree > max_value:
                    max_value = degree
                    class_value = rule.get_class()
        return class_value

    '''
       The granularity rules with normal rules, one new row data come, how to choose which rule
       Check if the data meet the granularity rule scope, if yes, go to the granularity rule, else
       go to the normal rules
    '''

    def FRM_Granularity(self, example):
        print("FRM_Granularity begin :  ")
        class_value = -1
        max_value = 0.0
        produc = 0
        for i in range(0, len(self.granularity_rule_Base)):
            rule = self.granularity_rule_Base[i]
            print("after get rule of the FRM_Granularity :")
            produc = rule.degree_product(example)
            print("produc  is ")
            print(produc)
            print("rule.wracc  is ")
            print(rule.wracc)
            produc *= rule.wracc
            if produc > max_value:
                max_value = produc
                class_value = rule.class_value
        if produc == 0:
            for i in range(0, len(self.rule_base_array)):
                rule = self.rule_base_array[i]
                produc = rule.degree_product(example)
                print("produc  is ")
                print(produc)
                print("rule.wracc  is ")
                print(rule.wracc)
                produc *= rule.wracc
                if produc > max_value:
                    max_value = produc
                    class_value = rule.class_value

        return class_value

    def FRM_Pruned_Granularity(self, example):
        # print("FRM_Granularity begin :  ")
        class_value = -1
        max_value = 0.0
        produc = 0
        for i in range(0, len(self.granularity_prune_rule_base)):
            rule = self.granularity_prune_rule_base[i]
            # print("after get rule of the FRM_Granularity :")
            produc = rule.compatibility(example)
            produc *= rule.weight
            if produc > max_value:
                max_value = produc
                class_value = rule.class_value

        return class_value

    # * Additive Combination FRM
    # * @param example double[] the input example
    # * @return int the class label for the set of rules with the highest sum of membership degree per class

    def frm_ac_with_two_parameters(self, example,selected_array):
        if self.train_myDataSet is None:
            class_value = None
            return class_value
        else:

            class_value = self.default_rule
            degree = Decimal(0.0)
            self.frm_ac_max_degree_value = Decimal(0.0)
            max_degree = Decimal(0.0)
            degrees_class = [0.0 for x in range(self.train_myDataSet.get_nclasses())]
            for i in range(0, self.train_myDataSet.get_nclasses()):
                degrees_class[i] = Decimal(0.0)
            rule_length =len(self.rule_base_array)
            if selected_array is None:
                selected_array= [1 for i in range(rule_length)]

            for i in range(0, len(self.rule_base_array)):
                if selected_array[i] > 0:
                    rule = self.rule_base_array[i]
                    degree = rule.matching(example)
                    degrees_class[rule.get_class()] += Decimal(degree)
            max_degree = 0.8
            sum_degree = Decimal(0.0)
            for i in range(0, self.train_myDataSet.get_nclasses()):
                sum_degree = sum_degree + degrees_class[i]
                if degrees_class[i] > max_degree:
                    max_degree = degrees_class[i]
                    self.frm_ac_max_degree_value = max_degree
                    class_value = i
            # print("the frm_ac_with_two_parameters return value is " + str(class_value))
            if sum_degree is 0:
                class_value = None

            return class_value

    def FRM_AC(self, example):

        degree = Decimal(0.0)
        self.frm_ac_max_degree_value = Decimal(0.0)
        class_value = self.default_rule

        degree_class_array = [0.0 for x in range(self.train_myDataSet.get_nclasses())]
        for i in range(0, self.train_myDataSet.get_nclasses()):
            degree_class_array[i] = Decimal(0.0)

        for i in range(0, len(self.rule_base_array)):
            rule = self.rule_base_array[i]

            degree = rule.matching(example)
            degree_class_array[rule.get_class()] += degree

        self.frm_ac_max_degree_value = 0.0
        for i in range(0, self.train_myDataSet.get_nclasses()):
            if degree_class_array[i] > self.frm_ac_max_degree_value:
                self.frm_ac_max_degree_value = degree_class_array[i]
                class_value = i

        return class_value

    # added by rui for negative  rules
    def generate_negative_rules(self, train, confident_value_pass, zone_confident_pass):

        class_value_arr = self.get_class_value_array(train)
        self.prepare_data_rows(train)
        for i in range(0, len(self.rule_base_array)):
            rule_negative = Rule(self.data_base)
            rule_negative.antecedent = self.rule_base_array[i].antecedent
            positive_rule_class_value = self.rule_base_array[i].get_class()
            print("the positive rule class value is " + str(positive_rule_class_value) + " ,the i is :" + str(i))
            # rule_negative.setClass(positive_rule_class_value)

            for j in range(0, len(class_value_arr)):
                class_type = int(class_value_arr[j])
                if positive_rule_class_value != class_type:  # need to get another class value for negative rule

                    rule_negative.setClass(class_type)  # change the class type in the rule
                    rule_negative.calculate_confident_support(self.data_row_array)
                    print("Negative rule's  confident value is :" + str(rule_negative.confident_value))

                    if rule_negative.confident_value > confident_value_pass and rule_negative.zone_confident > zone_confident_pass:
                        rule_negative.weight = rule_negative.confident_value
                        if not (self.duplicated_negative_rule(rule_negative)):

                            for k in range(0, len(rule_negative.antecedent)):
                                print("antecedent L_ " + str(rule_negative.antecedent[j]))
                            # print("Negative rule's class value " + str(rule_negative.get_class()))
                            # print(" Negative rule's weight, confident_vale  " + str(rule_negative.weight))
                            # print(" Negative rule's zone confident value   " + str(rule_negative.zone_confident))
                            # print("Negative rule's positive_rule_class_value" + str(positive_rule_class_value))
                            # print("Negative rule's class_type" + str(class_type))
                            self.negative_rule_base_array.append(rule_negative)

    def prepare_data_rows(self, train):
        for i in range(0, train.size()):
            data_row_temp = DataRow()
            class_value = train.get_output_as_integer_with_pos(i)
            example = train.get_example(i)
            example_feature_array = []
            example_feature_array.append(train.get_example(i))
            """
            for f_variable in range(0, self.n_variables):
                # print("The f_variable is :"+str(f_variable))
                # print("The example is :" + str(example))
                example_feature_array.append(train.get_example(f_variable))
            """

            label_array = []
            for m in range(0, self.n_variables):
                max_value = 0.0
                etq = -1
                per = None
                self.n_labels = self.data_base.num_labels(m)
                # print("n_labels: " + str(self.n_labels))
                for n in range(0, self.n_labels):
                    # print("Inside the second loop of searchForBestAntecedent......")
                    # print("example[" + str(m) + ")]: " + str(example[m]))
                    per = self.data_base.membership_function(m, n, example[m])
                    # print("per: " + str(per))
                    if per > max_value:
                        max_value = per
                        etq = n
                if max_value == 0.0:
                    # print("There was an Error while searching for the antecedent of the rule")
                    # print("Example: ")
                    for n in range(0, self.n_variables):
                        # print(str(example[n]) + "\t")
                        pass

                    # print("Variable " + str(m))
                    exit(1)
                # print(" The max_value is : " + str(max_value))
                # print(" ,the j value is : " + str(j))

                label_array.append(etq)

            data_row_temp.set_three_parameters(class_value, example_feature_array, label_array)
            self.data_row_array.append(data_row_temp)

    def get_class_value_array(self, train):
        class_value_array = []
        integer_array = train.get_output_as_integer()
        for i in range(0, len(integer_array)):
            exist_yes = False
            for j in range(0, len(class_value_array)):
                if integer_array[i] == class_value_array[j]:
                    exist_yes = True
            if not exist_yes:
                class_value_array.append(integer_array[i])
        return class_value_array

    def calculate_confident_support_rulebase(self, train):
        class_value_arr = self.get_class_value_array(train)
        str_print = "Totally there are: " + str(len(self.rule_base_array)) + " rules"
        # print(str_print)
        index_number = 1

        for each_rule in self.rule_base_array:
            each_rule.calculate_confident_support(self.data_row_array)
            # print(str(index_number) + " -- each_rule.weight :" + str(each_rule.weight) + ",zone_confident :" + str(
            #    each_rule.zone_confident) + ",calculate_confident :" + str(each_rule.confident_value))
            # print(" -- each_rule.support_value :" + str(each_rule.support_value))
            index_number = index_number + 1

    def get_inference_type(self):
        return self.inferenceType

    def get_k_value(self):
        return self.k_value

    """
   * Function to eliminate the rules that are not needed (Redundant, not enough accurate,...) for a given class.
   * @param clas class whose rules are being tested
    """

    def reduce_rules(self, class_value):
        nexamples = 0

        example_weight = []
        for i in range(0, self.train_myDataSet.size()):
            example_weight.append(ExampleWeight(self.k_value))
        selected = [0 for x in range(len(self.rule_base_array))]
        for i in range(0, len(self.rule_base_array)):
            selected[i] = 0

        nexamples = self.train_myDataSet.number_instances(class_value)
        nrule_select = 0
        posBestWracc = None

        while True:
            bestWracc = -1.0
            posBestWracc = -1
            for i in range(0, len(self.rule_base_array)):
                if selected[i] == 0:
                    rule = self.rule_base_array[i]
                    rule.calculate_wracc(self.train_myDataSet, example_weight)
                    if rule.get_wracc() > bestWracc:
                        bestWracc = rule.get_wracc()
                        posBestWracc = i
            if posBestWracc > -1:
                selected[posBestWracc] = 1
                nrule_select = nrule_select + 1
                rule = self.rule_base_array[posBestWracc]
                nexamples = nexamples - rule.reduce_weight(self.train_myDataSet, example_weight)

               
            if  (nexamples > 0 and (nrule_select < len(self.rule_base_array)) and (posBestWracc > -1)):
                pass
            else:
                break

        for i in range(len(self.rule_base_array)-1, -1, -1):
            if selected[i] == 0:
                self.rule_base_array.pop(i)
        #the last 0 is not considered by loop, so we must add it here.
        example_weight[:] = []
        # example_weight.clear()
        gc.collect()

    def add_rule(self, rule):
        self.rule_base_array.append(rule)

    def add_rule_base(self, rule_base_pass):

        for i in range(0, rule_base_pass.get_size()):
            self.rule_base_array.append(rule_base_pass.get(i).clone())

    """"
       * It adds a rule to the rule base
       * @param itemset itemset to be added
    """

    def add_itemset(self, itemset_pass):
        item = None
        antecedent_array = [0 for x in range(self.n_variables)]
        for i in range(0, self.n_variables):
            antecedent_array[i] = -1
        for i in range(0, itemset_pass.size()):
            item = itemset_pass.get(i)
            antecedent_array[item.get_variable()] = item.get_label()

        rule = Rule(self.data_base)
        rule.assign_antecedente(antecedent_array)
        rule.set_consequent(itemset_pass.get_class())
        rule.set_confidence(itemset_pass.get_support_class() / itemset_pass.get_support())
        rule.set_support(itemset_pass.get_support_class())
        self.rule_base_array.append(rule)

    def get_size(self):
        return len(self.rule_base_array)

    """
   * It removes the rule stored in the given position
   * @param pos Position where the rule we want to remove is
   * @return Removed rule
    """

    def remove(self, pos):
        return self.rule_base_array.pop(pos)

    def clear(self):
        self.rule_base_array[:] = []
        # self.rule_base_array.clear()
        self.fitness = 0.0

    """
     * Sets the default rule.
     * The default rule classifies all the examples to the majority class.
    """

    def set_default_rule(self):

        best_rule = 0
        for i in range(1, self.train_myDataSet.get_nclasses()):
            if self.train_myDataSet.number_instances(best_rule) < self.train_myDataSet.number_instances(i):
                best_rule = i
        self.default_rule = best_rule

    """

   * Function to return the fitness of the rule base

   * @return Fitness of the rule base
    
    """

    def get_accuracy(self):
        return self.fitness

    """
     * Indentifies how many classes are uncovered with a selection of rules.
     * @param selected rules selected to be tested
     * @return number of classes uncovered.
    """

    def has_class_uncovered(self, selected_array_pass):
        i = 0
        count = 0
        cover_array = []
        cover_array = [0 for x in range(self.train_myDataSet.get_nclasses())]
        for i in range(0, len(cover_array)):
            if self.train_myDataSet.number_instances(i) > 0:
                cover_array[i] = 0
            else:
                cover_array[i] = 1

        for i in range(0, len(self.rule_base_array)):
            if selected_array_pass[i] > 0:
                cover_array[self.rule_base_array[i].get_class()] += 1
        count = 0
        for i in range(0, len(cover_array)):
            if cover_array[i] == 0:
                count += 1

        return count

    """
   * Function to evaluate the whole rule base by using the training dataset.
     """

    def evaluate(self):
        nhits = 0
        prediction = 0

        self.nuncover = 0
        for j in range(0, self.train_myDataSet.get_nclasses()):
            self.nuncover_class_array[j] = 0

        for j in range(0, self.train_myDataSet.size()):
            prediction = self.frm(self.train_myDataSet.get_example(j))
            if self.train_myDataSet.get_output_as_integer_with_pos(j) == prediction:
                nhits += 1
            if prediction < 0:
                self.nuncover += 1
                self.nuncover_class_array[self.train_myDataSet.get_output_as_integer(j)] +=  1

        self.fitness = (100.0 * nhits) / (1.0 * self.train_myDataSet.size())
        # self.logger.debug("In evaluate of ruleBase , the self.fitness is :" + str(self.fitness))

    """
     * Function to evaluate the selected rules by using the training dataset and the fuzzy functions stored in the gene given.
     * @param gene Representation where the fuzzy functions needed to evaluate are stored
     * @param selected Selection of rules to be evaluated
     */
    """

    def evaluate_with_two_parameters(self, gene_array_pass, selected_array_pass):
        nhits = 0
        prediction = 0

        self.data_base.decode(gene_array_pass)

        nhits = 0
        self.nuncover = 0
        for i in range(0, self.train_myDataSet.get_nclasses()):
            self.nuncover_class_array[i] = 0

        for j in range(0, self.train_myDataSet.size()):

            prediction = self.frm_two_parameters(self.train_myDataSet.get_example(j), selected_array_pass)
            if self.train_myDataSet.get_output_as_integer_with_pos(j) == prediction:
                nhits += 1
            if prediction is None:
                print("Something wrong that the prediction is None")
            elif prediction < 0:
                self.nuncover += 1
                self.nuncover_class_array[self.train_myDataSet.get_output_as_integer_with_pos(j)] += 1
        train_size = self.train_myDataSet.size()
        if train_size is 0 or None:
            self.fitness = 0
        else:
            self.fitness = (100.0 * nhits) / (1.0 * self.train_myDataSet.size())
            self.logger.debug("In ruleBase , evaluate_with_two_parameters, recalulation the fitness, the self.fitness is :" + str(self.fitness))
        # print("evaluate_with_two_parameters :IN Rule Base, fitness is :" + str(self.fitness))

    """
     /**
     * Returns the number of examples uncovered by the rules
     * @return Number of examples uncovered
     */
    """

    def get_uncover(self):
        return self.nuncover

    """
    * Clone
    * @return A copy of the Rule Base
    """

    def clone(self):
        rule_base = RuleBase()
        rule_base.rule_base_array = []
        for i in range(0, len(self.rule_base_array)):
            rule_base.rule_base_array.append((self.rule_base_array[i]).clone())

        rule_base.data_base = self.data_base
        rule_base.train_myDataSet = self.train_myDataSet
        rule_base.n_variables = self.n_variables
        rule_base.fitness = self.fitness
        rule_base.K = self.k_value
        rule_base.inferenceType = self.inferenceType
        rule_base.default_rule = self.default_rule
        rule_base.nuncover = self.nuncover
        rule_base.nuncover_class_array = [0 for x in range(self.train_myDataSet.get_nclasses())]
        for i in range(0, self.train_myDataSet.get_nclasses()):
            rule_base.nuncover_class_array[i] = self.nuncover_class_array[i]

        return rule_base

    """
   * It stores the rule base in a given file
   * @param filename Name for the rulebase file
    """

    def save_file(self, file_name):

        string_out = self.printString()
        file = open(file_name, "w+")
        file.write(string_out)
        file.close()

        """
       * Function to get a rule from the rule base
       * @param pos Position in the rule base where the desired rule is stored
       * @return The desired rule
        """

    def get(self, pos):
        return self.rule_base_array[pos]

        """  
        /**
    * Maximization
    * @param a first number
    * @param b second number
    * @return boolean true if a is greater than b
    */
    
    """

    def better(self,a, b):
        if a > b:
            return True
        else:
            return False
