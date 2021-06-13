from decimal import Decimal
from DataRow import DataRow


# * <p>This class contains the structure of a Fuzzy Rule</p>
# *
# * @version 1.0
# * @since JDK1.5

class Rule:
    """
      int[] antecedent;
      int clas, nAnts;
      double conf, supp, wracc;
      DataBase dataBase;

    """

    # jave version has the below variables
    antecedent = []
    class_value = Decimal(0)
    nants = Decimal(0)
    wracc = Decimal(0.0)
    # added at 2020/06/25 to check positive rule confident
    confident_value = Decimal(0.0)
    # added at 2020/06/30 to check positive rule support
    support_value = Decimal(0.0)
    data_base = None

    weight = None
    compatibilityType = None

    # added by rui for negative rule
    rule_type = None
    rule_priority = None
    data_row_here = None
    granularity_sub_zone = None

    # In this fuzzy zone, the confident is supp(xUY)/supp(x)
    zone_confident = None
    supp_xy = None
    supp_x = None
    rule_index = None
    rule_information =""

    def __init__(self, data_base_pass):

        self.antecedent = [0 for x in range(data_base_pass.num_variables())]
        for i in range(0, len(self.antecedent)):
            # Don't care
            self.antecedent[i] = -1
        self.class_value = -1
        self.data_base = data_base_pass
        self.confident_value = 0.0
        self.support_value = 0.0
        self.nants = 0
        self.wracc = 0.0

        # print("__init__ of Rule")
        self.data_row_here = DataRow()
        self.rule_index = 0

    """
    * Clone
    * @return A copy of the rule
    """

    def clone(self):
        rule = Rule(self.data_base)
        rule.antecedent = [0 for x in range(len(self.antecedent))]
        for i in range(0, len(self.antecedent)):
            rule.antecedent[i] = self.antecedent[i]
            rule.class_value = self.class_value
            rule.data_base = self.data_base
            rule.confident_value = self.confident_value
            rule.support_value = self.support_value
            rule.nAnts = self.nants
            rule.wracc = self.wracc
            rule.rule_index = self.rule_index
            rule.supp_x = self.supp_x
            rule.supp_xy = self.supp_xy

        return rule

    """
    * It sets the antecedent of the rule
    * @param antecedent Antecedent of the rule
    """

    def assign_antecedente(self, antecedent_array):
        self.nants = 0
        for i in range(0, len(antecedent_array)):
            self.antecedent[i] = antecedent_array[i]
            if self.antecedent[i] > -1:
                self.nants += 1

    def degree_product(self, example):
        degree = Decimal(1.0)
        for i in range(0, len(self.antecedent)):
            if degree > 0.0:
                # for item in example:
                # print("item in example is  :" + str(item))

                # print("i is :"+ str(i)+" len(self.antecedent) : " + str(len(self.antecedent))+"len(example) : "+ str(len(example)))
                degree *= self.data_base.matching(i, self.antecedent[i], example[i])
                # print("In degree_product,the i is  "+str(i))

        return_value = degree * Decimal(self.confident_value)
        # print("return_value:" + str(return_value))
        return return_value

    """
    * Function to check if a given example matchs with the rule (the rule correctly classifies it)
    * @param example  Example to be classified
    * @return 0.0 = doesn't match, >0.0 = does.
    """

    def matching(self, example):
        return self.degree_product(example)

    """

    /**
       * It sets the confidence of the rule
       * @param conf Confidence to be set
    """

    def set_confidence(self, confident_value):
        self.confident_value = confident_value

    """
       * It sets the consequent of the rule
       * @param clas Class of the rule
    """

    def set_consequent(self, class_value):
        self.class_value = class_value

    """
    * It sets the support of the rule
    * @param supp  Support to be set
    """

    def set_support(self, supp):
        self.support_value = supp

    """
   * It returns the Confidence of the rule
   * @return Confidence of the rule
    """

    def get_confidence(self):
        return round(self.confident_value,4)

    """
       * It returns the Wracc of the rule
       * @return Wracc of the rule
    """

    def get_wracc(self):
        return round(self.wracc,4)

    """

   * It returns the support of the rule
   * @return Support of the rule
    """

    def get_support(self):
        return round(self.support_value,4)

    """ 
    /**
    * Calculate Wracc for this rule.
    * The value of the measure Wracc for this rule will be stored on the attribute "wracc".
      decreasing of wracc
    * @param train Training dataset
    * @param exampleWeight Weights of the patterns
    """

    def calculate_wracc(self, train_mydataset_pass, example_weight_array):
        i = 0
        n_c = Decimal(0.0)
        degree = Decimal(0.0)
        exmple_weight = None

        n_a = n_ac = Decimal(0.0)

        for i in range(0, train_mydataset_pass.size()):
            exmple_weight = example_weight_array[i]
            if exmple_weight.is_active():
                degree = self.matching(train_mydataset_pass.get_example(i))
                if degree > 0.0:
                    degree *= Decimal(exmple_weight.get_weight())
                    n_a += degree

                    if train_mydataset_pass.get_output_as_integer_with_pos(i) == self.class_value:
                        n_ac += degree
                        n_c += Decimal(exmple_weight.get_weight())


                elif train_mydataset_pass.get_output_as_integer_with_pos(i) == self.class_value:
                    n_c += Decimal(exmple_weight.get_weight())

        if (n_a < 0.0000000001) or (n_ac < 0.0000000001) or (n_c < 0.0000000001):
            self.wracc = Decimal(-1.0)
        else:
            self.wracc = (n_ac / n_c) * ((n_ac / n_a) - Decimal(train_mydataset_pass.frecuent_class(self.class_value)))
        self.wracc = round(self.wracc, 4)
        print("self.wracc" + str(self.wracc))

    """

     * Reduces the weight of the examples that match with the rule (the rule correctly classifies them)
     * @param train training examples given to match them to the rule.
     * @param exampleWeight Each example weight to be updated.
     * @return Number of examples that have become not active after the weight reduction.
     */

    """

    def reduce_weight(self, train_mydataset_pass, example_weight_array):
        count = 0
        for i in range(0, train_mydataset_pass.size()):
            example_weight = example_weight_array[i]
            if example_weight.is_active():
                if self.matching(train_mydataset_pass.get_example(i)) > 0.0:
                    example_weight.inc_count()
                    if (not example_weight.is_active()) and (
                            train_mydataset_pass.get_output_as_integer_with_pos(i) == self.class_value):
                        count = count + 1
        return count

    # not exist in the java version below

    def setClass(self, clas):
        self.class_value = clas

    # added by rui for negative rule
    def get_class(self):
        return self.class_value

    # * Operator T-min
    # * @param example double[] The input example
    # * @return double the computation the the minimum T-norm
    """

    def minimumCompatibility(self, example):
        minimum = None
        membershipDegree = None
        minimum = 1.0
        for i in range(0, len(self.antecedent)):
            # print("example[" + str(i) + "] = " + example[i])
            membershipDegree = self.antecedent[i].fuzzify(example[i])
            # print("membershipDegree in minimumCompatibility = " + str(membershipDegree))
            minimum = min(membershipDegree, minimum)

        return minimum
           """

    # * Operator T-product
    # * @param example double[] The input example
    # * @return double the computation the the product T-norm
    # arrive here
    """
    def productCompatibility(self, example):

        product = 1.0
        antecedent_number = len(self.antecedent)
        # print("antecedent_number = " + str(antecedent_number))
        # print("before the antecedent loop :")
        for i in range(0, antecedent_number):
            # print("example[i="+ str(i)+"]"+":"+ str(example[i]))
            # print("in loop before get memebershipdegree")
            membershipDegree = self.antecedent[i].fuzzify(example[i])
            # print("membershipDegree in productCompatibility  = " +str(membershipDegree))
            product = product * membershipDegree
        # print("product: "+ str(product))
        product = round(product, 4)
        return product
           

    # * Classic Certainty Factor weight
    # * @param train myDataset training dataset
 
    def consequent_CF(self, train):
        train_Class_Number = train.getnClasses()
        # to have enough class_sum space
        classes_sum = [0.0 for x in range(train_Class_Number + 1)]
        for i in range(0, train.getnClasses() + 1):
            classes_sum[i] = 0.0

        total = 0.0
        comp = None
        # Computation of the sum by classes */
        for i in range(0, train.size()):
            comp = self.compatibility(train.getExample(i))
            classes_sum[train.getOutputAsIntegerWithPos(i)] = classes_sum[train.getOutputAsIntegerWithPos(i)] + comp
            total = total + comp

        # print("classes_sum[self.class_value]  = " + str(classes_sum[self.class_value]) + "total" + str(total))
        self.weight = round((classes_sum[self.class_value] / total), 4)

    # * Penalized Certainty Factor weight II (by Ishibuchi)
    # * @param train myDataset training dataset
   

    

    def consequent_PCF2(self, train):
        classes_sum = float[train.getnClasses()]
        for i in range(0, train.getnClasses()):
            classes_sum[i] = 0.0

        total = 0.0
        comp = None
        # Computation of the sum by classes */
        for i in range(0, train.size()):
            comp = self.compatibility(train.getExample(i))
            classes_sum[train.getOutputAsIntegerWithPos(i)] = classes_sum[train.getOutputAsIntegerWithPos(i)] + comp
            total = total + comp

        sum_value = (total - classes_sum[self.class_value]) / (train.getnClasses() - 1.0)
        self.weight = round(((classes_sum[self.class_value] - sum_value) / total), 4)

    # * Penalized Certainty Factor weight IV (by Ishibuchi)
    # * @param train myDataset training dataset

    def consequent_PCF4(self, train):
        class_number = train.getnClasses()
        # print("train data set get the class_number: " + str(class_number))
        classes_sum_number = class_number + 1
        classes_sum = [0.0 for x in range(classes_sum_number)]
        # print("classes_sum length is : " + str(len(classes_sum)))
        # for have enough classes_sum for class value
        for i in range(0, train.getnClasses() + 1):
            classes_sum[i] = 0.0

        total = 0.0
        train_size = train.size()
        # print("train_size: " + str(train_size))
        # Computation of the sum by classes */
        # print("Begin a new loop for calculating comp " + "/n/n")
        zeroCompNumber = 0

        # for i in range(0, train_size):
        # print("train.getExample(i) : " + str(train.getExample(i)))
        # class_type = train.getOutputAsIntegerWithPos(i)
        # print("test the class type print is : " + str(class_type))

        for i in range(0, train_size):
            # print("train.getExample(i) : " + str(train.getExample(i)))
            comp = self.compatibility(train.getExample(i))
            if comp == 0:
                zeroCompNumber = zeroCompNumber + 1

            # print(" The list index out of range is i = " + str(i))
            class_type = train.getOutputAsIntegerWithPos(i)
            # print(" class_type = " + str(class_type))
            classes_sum[class_type] = classes_sum[class_type] + comp
            total = total + comp

        # print("self.clas =" + str(self.class_value) + "classes_sum[self.clas] :" + str(classes_sum[self.class_value]))
        # print(" The zero comp number in this loop is :" + str(zeroCompNumber))
        sum_value = total - classes_sum[self.class_value]
        self.weight = round(((classes_sum[self.class_value] - sum_value) / total), 4)
        # print("self.weight is " + str(self.weight))

    # * This function detects if one rule is already included in the Rule Set
    # * @param r Rule Rule to compare
    # * @return boolean true if the rule already exists, else false

    def comparison(self, rule):
        contador_value = 0
        for j in range(0, len(self.antecedent)):
            if self.antecedent[j] == rule.antecedent[j]:
                contador_value = contador_value + 1

        if contador_value == len(rule.antecedent):
            if self.class_value != rule.class_value:  # Comparison of the rule weights
                if self.weight < rule.weight:
                    # Rule Update
                    self.class_value = rule.class_value
                    self.weight = rule.weight

            return True
        else:
            return False
"""
    def calculate_confident_support(self, data_row_array):
        # how many instances in the zone
        supp_x = 0
        # instances in the zone with the same expected class value
        supp_xy = 0
        self.confident_value = 0
        all_number_of_the_class = 0
        total_number = len(data_row_array)
        for i in range(0, total_number):
            self.data_row_here = data_row_array[i]
            #  print("self.data_row_here.class_value  :" + str(self.data_row_here.class_value))
            #  print("self.class_value  :" + str(self.class_value))
            if self.data_row_here.class_value == self.class_value:
                all_number_of_the_class = all_number_of_the_class + 1
            meet_antecedent = 0

            for j in range(0, len(self.data_row_here.label_values)):
                # print("******self.antecedent[j]  :" + str(self.antecedent[j]))
                # print("self.self.data_row_here.label_values[j]  :" + str(self.data_row_here.label_values[j])+"*******")

                if self.antecedent[j] == self.data_row_here.label_values[j]:  # meet the rule antecedent conditions
                    meet_antecedent = meet_antecedent + 1
            if self.get_antecident_number(self.antecedent) == meet_antecedent:
                supp_x = supp_x + 1
                if self.data_row_here.class_value == self.class_value:
                    supp_xy = supp_xy + 1

        if all_number_of_the_class != 0:
            # print("support_rule_number :"+str(support_rule_number))
            # print("all_number_of_the_class :" + str(all_number_of_the_class))
            self.support_value = round((supp_x / total_number), 4)
            # print("self.support_value in the rule:" + str(self.support_value))
            self.confident_value = round((supp_xy / all_number_of_the_class), 4)
            # print("self.confident_value in the rule:" + str(self.confident_value))
        if supp_x != 0:
            self.zone_confident = round((supp_xy / supp_x), 4)
        self.supp_x = supp_x
        self.supp_xy = supp_xy

    def print_rule_information(self,n_variables,train_myDataSet):

        self.rule_information =""
        names =train_myDataSet.get_names()
        classes = train_myDataSet.get_classes()

        for j in range(0, n_variables):
            if self.antecedent[j] < 0:
                pass
            else:
                if j < n_variables and self.antecedent[j] >= 0:
                    self.rule_information += names[j] + " IS " + self.data_base.print_here(j, self.antecedent[j]) + " AND "


        self.rule_information += ": " + classes[self.class_value]
        self.rule_information += " CF: " + str(self.get_confidence()) + "\n"

        self.rule_information = self.rule_information + " /n "+ "how many instance number covered by the rule :" + str(self.supp_xy)+ "  "
        self.rule_information = self.rule_information + " /n "+  "how many instance number only antecedent covered by the rule :" + str(self.supp_x)+ "  "

        return self.rule_information


    def get_antecident_number(self, antecedent_array):
        antecedent_number = 0
        for i in range(0, len(antecedent_array)):
            if not antecedent_array[i] == -1:
                antecedent_number = antecedent_number + 1

            return antecedent_number

    def assing_consequent(self, train):
        pass
