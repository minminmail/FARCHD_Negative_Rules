# '''
#  * <p>It contains the methods to read a Classification/Regression Dataset</p>
#  *
#  * @version 1.1
#  * @since JDK1.5
# '''
from decimal import Decimal

from Help_Classes.InstanceSet import InstanceSet
from Help_Classes.Attributes import Attributes
from Help_Classes.Attribute import Attribute
import math
import numpy as np


class MyDataSet:
    # Number to represent type of variable real or double.
    REAL = 0
    # *Number to represent type of variable integer.*
    INTEGER = 1
    # *Number to represent type of variable nominal.*
    NOMINAL = 2

    x_array = []  # examples array
    missing_array = []  # possible missing values
    output_integer_array = []  # output of the data - set as integer values private
    output_real_array = []  # output of the data - set as double values
    output_array = []  # output of the data - set as string values
    emax_array = []  # max value of an attribute private
    emin_array = []  # min value of an attribute

    ndata = None  # Number of examples
    nvars = None  # Numer of variables
    ninputs = None  # Number of inputs
    nclasses = None  # Number of outputs

    instance_set = None  # The whole instance set
    stdev_array = []
    average_array = []  # standard deviation and average of each attribute
    instances_cl = []

    # nominal  attributes bool array
    nominal_array = []
    # integer   attributes int array
    integer_array = []

    frequent_class_array = []

    #  *Init a new set of instances

    def __init__(self):
        self.instance_set = InstanceSet()

    # '''
    #    * Outputs an array of examples with their corresponding attribute values.
    #    * @return double[][] an array of examples with their corresponding attribute values
    #  '''
    def get_x(self):
        return self.x_array

    def set_x(self, x_parameter):
        self.x_array = x_parameter

    # '''
    #    * Output a specific example
    #    * @param pos int position (id) of the example in the data-set
    #    * @return double[] the attributes of the given example
    # '''
    def get_example(self, pos):
        # # print(" In getExample, len(self.x_array) = " + str(len(self.x_array)) + ", pos = " + str(
        #   pos) + "  ," + "self.x_array[pos] ==" + str(self.x_array[pos]))
        return self.x_array[pos]

    # * Returns the output of the data-set as integer values
    # * @return int[] an array of integer values corresponding to the output values of the dataset

    def get_output_as_integer(self):
        size = len(self.output_integer_array)
        output = [0 for x in range(size)]
        for i in range(0, size):
            output[i] = self.output_integer_array[i]
        return output

    #    * Returns the output of the data-set as real values
    #    * @return double[] an array of real values corresponding to the output values of the dataset

    def get_output_as_real(self):
        output_length = len(self.output_real_array)
        output = [0.0 for x in range(output_length)]
        for i in range(0, len(self.output_real_array)):
            output[i] = self.output_integer_array[i]
        return output

    #    * Returns the output of the data-set as nominal values
    #    * @return String[] an array of nomianl values corresponding to the output values of the dataset
    #

    def get_output_as_string(self):
        output_length = len(self.output_array)
        output = ["" for x in range(output_length)]
        for i in range(0, output_length):
            output[i] = self.output_array[i]

        return output

    #    * It returns the output value of the example "pos"
    #    * @param pos int the position (id) of the example
    #    * @return String a string containing the output value

    def get_output_as_string_with_pos(self, pos):
        # # print("pos is in getOutputAsStringWithPos "+str(pos))
        # maybe the exception is here.
        return self.output_array[pos]

    #    * It returns the output value of the example "pos"
    #    * @param pos int the position (id) of the example
    #    * @return int an integer containing the output value

    def get_output_as_integer_with_pos(self, pos):
        return self.output_integer_array[pos]

    def set_output_integer_array(self, integer_array):
        self.output_integer_array = integer_array

    def set_output_array(self, output_array):
        self.output_array = output_array

    #    * It returns the output value of the example "pos"
    #    * @param pos int the position (id) of the example
    #    * @return double a real containing the output value

    def get_output_as_real_with_pos(self, pos):
        return self.output_real_array[pos]

        # *It returns an array with the maximum values of the attributes
        # * @ return double[] an array with the maximum values of the attributes
        #

    def get_emax(self):
        return self.emax_array

        # *It returns an array with the minimum values of the attributes
        # * @ return double[] an array with the minimum values of the attributes

    def get_emin(self):
        return self.emin_array

    # *It returns the maximum value of the given attribute
    # *
    # * @ param variable the index of the attribute
    # * @ return the maximum value of the given attribute

    def get_max(self, variable):
        return self.emax_array[variable]

    # *It returns the minimum value of the given attribute
    #
    # * @ param variable the index of the attribute
    # * @ return the minimum value of the given attribute

    def get_min(self, variable):
        return self.emin_array[variable]

    # *It gets the size of the data - set
    # * @ return int the number of examples in the data - set

    def get_ndata(self):

        return self.ndata

    def set_ndata(self, ndata):
        self.ndata = ndata

    # *It gets the number of variables of the data - set(including the output)
    # * @ return int the number of variables of the data - set(including the output)

    # modified at 2020-08-14
    def get_nvars(self):
        return self.nvars

    #    * It gets the number of input attributes of the data-set
    #    * @return int the number of input attributes of the data-set

    def get_ninputs(self):
        return self.ninputs

    def set_ninputs(self, ninputs_value):
        self.ninputs = ninputs_value

    #    * It gets the number of output attributes of the data-set (for example number of classes in classification)
    #    * @return int the number of different output values of the data-set

    def get_nclasses(self):
        return self.nclasses

    def set_nclasses(self, nclasses_value):
        self.nclasses = nclasses_value

    # added by rui for granularity rule generation
    def calculate_nclasses_for_small_granularity_zone(self, output_integer_array):
        class_number = 0
        class_array = []
        has_class = False
        for i in range(0, len(output_integer_array)):
            # # print(" output_integer_array[i] " + str(output_integer_array[i]))
            if len(class_array) == 0:
                class_array.append(output_integer_array[i])
            else:
                has_class = False
                for j in range(0, len(class_array)):
                    if class_array[j] == output_integer_array[i]:
                        # # print(" class_array[j] " + str(class_array[j]))
                        has_class = True
                if not has_class:
                    class_array.append(output_integer_array[i])
        class_number = len(class_array)
        return class_number

    #  * This function checks if the attribute value is missing
    #  * @param i int Example id
    #  * @param j int Variable id
    #  * @return boolean True is the value is missing, else it returns false

    def is_missing(self, i, j):
        return self.missing_array[i][j]

    #  * It reads the whole input data-set and it stores each example and its associated output value in
    #  * local arrays to ease their use.
    #  * @param datasetFile String name of the file containing the dataset
    #  * @param train boolean It must have the value "true" if we are reading the training data-set
    #  * @throws IOException If there ocurs any problem with the reading of the data-set

    def read_classification_set(self, dataset_file, train, file_path):
        try:
            # Load in memory a dataset that contains a classification problem
            print("Inside read_classification_set, datasetFile :" + str(dataset_file))
            # print("train is :" + str(train))
            # print("object instanceSet is :" + str(self.instance_set))

            if self.instance_set is None:
                print("self.instance_set is Null")
            else:
                no_outputs = None
                print("self.instance_set is not None, train = " + str(train))
                self.instance_set.read_set(dataset_file, train, file_path)
                print("begin getNumInstances ...... in read_classification_set ")
                self.ndata = self.instance_set.getNumInstances()
                print("In readCread_classification_setlassificationSet , self.ndata is : " + str(self.ndata))
                self.ninputs = Attributes.getInputNumAttributes(Attributes)
                print("In read_classification_set , self.ninputs is : " + str(self.ninputs))
                self.nvars = self.ninputs + Attributes.getOutputNumAttributes(Attributes)
                print("In read_classification_set , self.nvars is : " + str(self.nvars))

                # outputInteger check that there is only one output variable
                if Attributes.getOutputNumAttributes(Attributes) > 1:
                    outAttrs = Attributes.getOutputAttributes(Attributes)
                    # print("Output Attributes number is bigger than 1")
                    i = 1
                    for outAtt in outAttrs:
                        # print("Att" + str(i) + str(outAtt.getName()))
                        i = i + 1
                    # print("" + Attributes.getOutputAttributesHeader(Attributes))
                    print("This algorithm can not process MIMO datasets !!! exit 1")
                    # print("All outputs but the first one will be removed")
                    exit(1)
                no_outputs = False
                if Attributes.getOutputNumAttributes(Attributes) < 1:
                    print("This algorithm can not process datasets without outputs !!!!!!")
                    # print("Zero-valued output generated")
                    no_outputs = True
                    exit(1)

                # print("define all the array in MyDataSet class......")
                # Initialice and fill our own tables
                # print("The two dimension array X, dimension 1 is :" + str(self.ndata) + " ,Dimension 2 is :" + str(self.ninputs))

                ndata_length = self.ndata
                ninput_length = self.ninputs
                print("nDataLength = " + str(ndata_length))
                # print("nInputLength = " + str(nInputLength))
                # [[0 for j in range(m)] for i in range(n)] first column, then row

                self.x_array = [[0.0 for y in range(ninput_length)] for x in range(ndata_length)]

                self.missing_array = [[True for y in range(ninput_length)] for x in range(ndata_length)]

                self.nominal_array = [True for x in range(ninput_length)]
                self.integer_array = [True for x in range(ninput_length)]

                self.output_integer_array = [0 for x in range(ndata_length)]

                self.output_real_array = [0.0 for x in range(ndata_length)]
                self.output_array = ["" for x in range(ndata_length)]

                # Maximum and minimum of inputs
                self.emax = [0.0 for x in range(ninput_length)]
                self.emin = [0.0 for x in range(ninput_length)]

                for i in range(0, ninput_length):

                    attribute_instance = Attributes.getInputAttribute(Attributes, i)

                    if attribute_instance.getNumNominalValues() > 0:
                        self.emin[i] = 0
                        self.emax[i] = Attributes.getInputAttribute(i).getNumNominalValues() - 1
                    else:
                        self.emax[i] = Attributes.getAttributeByPos(Attributes, i).getMaxAttribute()
                        self.emin[i] = Attributes.getAttributeByPos(Attributes, i).getMinAttribute()

                    if attribute_instance.getType() == Attribute.NOMINAL:
                        self.nominal_array[i] = True
                        self.integer_array[i] = False
                    elif attribute_instance.getType() == Attribute.INTEGER:
                        self.nominal_array[i] = False
                        self.integer_array[i] = True
                    else:
                        self.nominal_array[i] = False
                        self.integer_array[i] = False

                    # print("self.emax[n]:" + str(self.emax[n]))
                    # print("self.emin[n]:" + str(self.emin[n]))
                # All values are casted into double/integer

                self.nclasses = 0
                for i in range(0, ndata_length):
                    inst = self.instance_set.getInstance(i)
                    for j in range(0, ninput_length):
                        input_Numeric_Value = self.instance_set.getInputNumericValue(i, j)
                        # # print("self.x_array [i] = " + str(i) + ",[j] = " + str(j) + ",input_Numeric_Value:" + str(
                        #  input_Numeric_Value))

                        self.x_array[i][j] = input_Numeric_Value  # inst.getInputRealValues(j);
                        # # print("after get self.x_array[i][j]")
                        self.missing_array[i][j] = inst.getInputMissingValuesWithPos(j)
                        # # print("after self.missing_array[i][j]")
                        if self.missing_array[i][j]:
                            self.x_array[i][j] = self.emin[j] - 1

                    if no_outputs:
                        # print("no_outputs==True")
                        self.output_integer_array[i] = 0
                        # elf.output_real_array[i] = 0.0
                        self.output_array[i] = ""
                    else:
                        # print("no_outputs==False")
                        self.output_integer_array[i] = self.instance_set.getOutputNumericValue(i, 0)
                        # print(" 202001-1 self.output_integer_array[ "+str(i)+"]"+ str( self.output_integer_array[i]))
                        # self.output_real_array[i] = self.instance_set.getOutputNumericValue(i, 0)
                        # print("self.output_integer_array[" + str(i) + "] = " + str(self.output_integer_array[i]))
                        self.output_array[i] = self.instance_set.getOutputNominalValue(i, 0)
                    # print(" 202001-1 self.output_integer_array[ " + str(i) + "]" + str(self.output_integer_array[i]))
                    if self.output_integer_array[i] > self.nclasses:
                        self.nclasses = self.output_integer_array[i]

                self.nclasses = self.nclasses + 1
                print('Number of classes=' + str(self.nclasses))
        except Exception as error:
            print("read_classification_set: Exception in readSet, in read_classification_set:" + str(error))

        # self.computeStatistics()
        self.compute_instances_per_class()

    #   * It reads the whole input data-set and it stores each example and its associated output value in
    #   * local arrays to ease their use.
    #   * @param datasetFile String name of the file containing the dataset
    #   * @param train boolean It must have the value "true" if we are reading the training data-set
    #   * @throws IOException If there ocurs any problem with the reading of the data-set

    # added by rui for granularity rule generation
    def read_classification_set_from_data_row_array(self, data_row_array):

        self.compute_statistics_data_row_array(data_row_array)
        self.compute_instances_perclass_data_row_array(data_row_array)

    def readRegressionSet(self, datasetFile, train, file_path):

        try:
            # Load in memory a dataset that contains a regression problem
            self.instance_set.readSet(datasetFile, train, file_path)
            self.ndata = self.instance_set.getNumInstances()
            self.ninputs = Attributes.getInputNumAttributes(Attributes)
            self.nvars = self.ninputs + Attributes.getOutputNumAttributes(Attributes)
            # print("In readRegressionSet , self.ndata is : " + str(self.ndata))
            # print("In readRegressionSet , self.ninputs is : " + str(self.ninputs))
            # print("In readRegressionSet , self.nvars is : " + str(self.nvars))

            # outputIntegerheck that there is only one output variable
            if Attributes.getOutputNumAttributes(Attributes) > 1:
                # print("Out put attribute: ")
                outPutAttHeader = Attributes.getOutputAttributesHeader(Attributes)
                # print(outPutAttHeader)
                # print("This algorithm can not process MIMO datasets")
                # print("All outputs but the first one will be removed")
                exit(1)

            noOutputs = False
            if Attributes.getOutputNumAttributes(Attributes) < 1:
                # print("This algorithm can not process datasets without outputs")
                # print("Zero-valued output generated")
                noOutputs = True
                print("noOutputs = True, exit 1 !!!!!")
                exit(1)
            # Initialice and fill our own tables
            self.x_array = [[0.0 for y in range(self.ninputs)] for x in range(self.ndata)]
            self.missing_array = [[False for y in range(self.ninputs)] for x in range(self.ndata)]
            self.output_integer_array = [0 for x in range(self.ndata)]

            # Maximum and minimum of inputs
            self.emax_array = [None for x in range(self.ninputs)]
            self.emin_array = [None for x in range(self.ninputs)]
            for i in range(0, self.ninputs):
                self.emax_array[i] = Attributes.getAttributeByPos(Attributes, i).getMaxAttribute()
                self.emin_array[i] = Attributes.getAttributeByPos(Attributes, i).getMinAttribute()

            # All values are casted into double / integer
            self.nclasses = 0

            for i in range(0, self.ndata):
                inst = self.instance_set.getInstance(i)
                for j in range(0, self.ninputs):
                    self.x_array[i][j] = self.instance_set.getInputNumericValue(i, j)  # inst.getInputRealValues(j);
                    self.missing_array[i][j] = inst.getInputMissingValues(j)
                    if self.missing_array[i][j]:
                        self.x_array[i][j] = self.emin_array[j] - 1

                if noOutputs:
                    print("noOutputs self.output_real_array[i]" + str(i) + "is 0 ")
                    self.output_real_array[i] = 0

                    self.output_integer_array[i] = 0

                else:
                    print("noOutputs else part:")

                    self.output_real_array[i] = self.instance_set.getOutputNumericValue(i, 0)
                    print("self.output_real_array[i]" + str(i) + str(self.output_real_array[i]))
                    self.output_integer_array[i] = int(self.output_real_array[i])
        except OSError as error:
            print("OS error: {0}".format(error))
        except Exception as otherException:
            # print("DBG: Exception in readSet:", sys.exc_info()[0])
            print(" In readRegressionSet other Exception  is :" + str(otherException))

        self.computeStatistics()

    # *It copies the header of the dataset
    # * @ return String A string containing all the data - set information

    def copy_header(self):

        p = ""
        # # print("copyHeader begin...., P is :" + p)
        p = "@relation " + Attributes.getRelationName(Attributes) + "\n"
        # # print(" after relation P is :" + p)
        p += Attributes.getInputAttributesHeader(Attributes)
        # # print(" after getInputAttributesHeader P is :" + p)
        p += Attributes.getOutputAttributesHeader(Attributes)
        # # print(" after getOutputAttributesHeader P is :" + p)
        p += Attributes.getInputHeader(Attributes) + "\n"
        # # print(" after getInputHeader P is :" + p)
        p += Attributes.getOutputHeader(Attributes) + "\n"
        # # print(" after getOutputHeader P is :" + p)
        p += "@data\n"

        # print("P is :" + p)
        return p

    #    * It transform the input space into the [0,1] range

    def normalize(self):
        atts = self.getn_inputs()
        maxs = [0.0 for x in range(atts)]
        for j in range(0, atts):
            maxs[j] = 1.0 / (self.emax_array[j] - self.emin_array[j])

        for i in range(0, self.get_ndata()):
            for j in range(0, atts):
                if not self.isMissing(i, j):  # this process ignores missing values
                    self.x_array[i][j] = (self.x_array[i][j] - self.__emin[j]) * maxs[j]

    # * It checks if the data-set has any real value
    # * @return boolean True if it has some real values, else false.

    def has_real_attributes(self):
        return Attributes.hasRealAttributes(self)

    #    * It checks if the data-set has any real value
    #    * @return boolean True if it has some real values, else false.

    def has_numerical_attributes(self):
        return Attributes.hasIntegerAttributes(self) or Attributes.hasRealAttributes(self)

    #    * It checks if the data-set has any missing value
    #    * @return boolean True if it has some missing values, else false.

    def has_missing_attributes(self):
        return self.size_without_missing() < self.get_ndata()

    #    * It return the size of the data-set without having account the missing values
    #    * @return int the size of the data-set without having account the missing values

    def size_without_missing(self):
        tam = 0
        # # print("self.ndata is :" + str(self.ndata) + ", self.ninputs :" + str(self.ninputs))
        for i in range(0, self.ndata):
            for j in range(1, self.ninputs):
                # changed the isMissing condition inside if
                if self.is_missing(i, j):
                    # print("It is missing value is i = " + str(i) + ",j==" + str(j))
                    break
            j = j + 1
            # # print("sizeWithoutMissing,  i = " + str(i) + ",j==" + str(j))
            if j == self.ninputs:
                tam = tam + 1
        # print("tam=" + str(tam))
        return tam

    #    * It returns the number of examples
    #    *
    #    * @return the number of examples

    def size(self):
        return self.ndata

    #    * It computes the average and standard deviation of the input attributes

    def compute_statistics(self):
        try:
            print("Begin computeStatistics......")
            var_num = self.get_nvars()
            print("varNum = " + str(var_num))
            self.stdev_array = [0.0 for x in range(var_num)]  # original was double ,changed into float in python
            self.average_array = [0.0 for x in range(var_num)]

            input_num = self.getn_inputs()
            data_num = self.get_ndata()
            print("inputNum = " + str(input_num) + ",dataNum = " + str(data_num))
            for i in range(0, input_num):
                self.average_array[i] = 0
                for j in range(0, data_num):
                    if not self.isMissing(j, i):
                        self.average_array[i] = self.average_array[i] + self.x_array[j][i]
                if data_num != 0:
                    self.average_array[i] = self.average_array[i] / data_num
            average_length = len(self.average_array)
            print(" average_length is " + str(average_length))
            self.average_array[average_length - 1] = 0
            if len(self.output_real_array) == 0:
                print("len(self.output_real_array) is  0")

            else:
                # print("len(self.output_real_array) is " + str(len(self.output_real_array)))
                for j in range(0, len(self.output_real_array)):
                    # print("self.output_real_array[j] is : "+str(self.output_real_array[j]) + " ,j is :"+str(j))
                    self.average_array[average_length - 1] = self.average_array[average_length - 1] + \
                                                             self.output_real_array[j]
            if len(self.output_real_array) != 0:
                self.average_array[average_length - 1] = self.average_array[average_length - 1] / len(
                    self.output_real_array)
                print("before the loop for inputNum")
                for i in range(0, input_num):
                    sum_value = 0.0
                    for j in range(0, data_num):
                        if not self.isMissing(j, i):
                            # print("self.isMissing(j, i)==False")
                            sum_value = sum_value + (self.x_array[j][i] - self.average_array[i]) * (
                                    self.x_array[j][i] - self.average_array[i])

                    if data_num != 0:
                        # print("dataNum != 0" + " , dataNum=" + str(data_num))
                        sum_value = sum_value / data_num
                    self.stdev_array[i] = math.sqrt(sum_value)

                sum_value = 0.0
                for j in range(0, len(self.output_real_array)):
                    sum_value += (self.output_real_array[j] - self.average_array[average_length - 1]) * (
                            self.output_real_array[j] - self.average_array[average_length - 1])
                if len(self.output_real_array) != 0:
                    sum_value /= len(self.output_real_array)
                self.stdev_array[len(self.stdev_array) - 1] = math.sqrt(sum_value)
                print("sum is :" + str(sum_value) + "  self.stdev_array :" + str(self.stdev_array))
        except Exception as error:
            print("Exception in computeStatistics : " + str(error))

    #    * It return the standard deviation of an specific attribute
    #    * @param position int attribute id (position of the attribute)
    #    * @return double the standard deviation  of the attribute

    def std_dev(self, position):
        return self.stdev_array[position]

    #    * It return the average of an specific attribute
    #    * @param position int attribute id (position of the attribute)
    #    * @return double the average of the attribute

    def average(self, position):
        return self.average_array[position]

    #     *It computes the number of examples per class

    def compute_instances_per_class(self):
        # print("compute_instances_per_class begin..., self.nclasses = " + str(self.nclasses))
        self.instances_cl = [0 for x in range(self.nclasses)]
        self.frequent_class_array = [Decimal(0.0) for x in range(self.nclasses)]
        data_num = self.get_ndata()
        # print("dataNum = " + str(dataNum))

        for i in range(0, data_num):
            integer_in_loop = self.output_integer_array[i]
            # # print("outputInteger[" + str(i) + "]" + str(integerInLoop))
            self.instances_cl[integer_in_loop] = self.instances_cl[integer_in_loop] + 1

        for i in range(0, self.nclasses):
            if data_num is 0:
                self.frequent_class_array[i] = 0
            else:
                self.frequent_class_array[i] = (1.0 * self.instances_cl[i] / data_num)

    #     *It returns the number of examples for a given class
    #     * @ param clas int the class label id
    #     * @ return int the number of examples
    #     for the class

    def number_instances(self, clas):
        return self.instances_cl[clas]

    # /**
    #  * It returns the number of labels for a nominal attribute
    #  * @param attribute int the attribute position in the data-set
    #  * @return int the number of labels for the attribute
    #  */
    #

    def number_values(self, attribute):
        return Attributes.getInputAttribute(attribute).getNumNominalValues(Attributes)

    #    * It returns the class label (string) given a class id (int)
    #    * @param intValue int the class id
    #    * @return String the corrresponding class label
    #

    #    * It returns the class label (string) given a class id (int)
    #    * @param intValue int the class id
    #    * @return String the corrresponding class label

    def get_output_value(self, int_value):
        # # print("Before att get ")
        att = Attributes.getOutputAttribute(Attributes, 0)
        # # print("After att get ")
        return att.getNominalValue(int_value)

    #  * It returns the type of the variable
    #  * @param variable int the variable id
    #  * @return int a code for the type of the variable (INTEGER, REAL or NOMINAL)

    def get_type(self, variable):
        if Attributes.getAttributeByPos(variable).getType() == Attributes.getAttributeByPos(Attributes, 0).INTEGER:
            return self.INTEGER

        if Attributes.getAttributeByPos(variable).getType() == Attributes.getAttributeByPos(Attributes, 0).REAL:
            return self.REAL

        if Attributes.getAttributeByPos(variable).getType() == Attributes.getAttributeByPos(Attributes, 0).NOMINAL:
            return self.NOMINAL

        return 0

    #  * It returns the discourse universe for the input and output variables
    #  * @return double[][] The minimum [0] and maximum [1] range of each variable
    def set_nvars(self, nvar_value):
        self.nvars = nvar_value

    # modified at 2020-08-14
    def get_ranges(self):

        # print("self.get_nvars()" + str(self.get_nvars()))
        rangos = [[0.0 for y in range(2)] for x in range(self.get_nvars())]
        # print("rangos has two dimensions, first is self.get_nvars()==" + str(self.getn_inputs()) + ",second is 2")
        ninputs = self.get_ninputs()
        for i in range(0, ninputs):
            # print("self.getn_inputs() is :" + str(nInputs) + " i = " + str(i))
            attHere = Attributes.getInputAttribute(Attributes, i)
            # print("attHere.getNumNominalValues()== " + str(attHere.getNumNominalValues()))
            if attHere.getNumNominalValues() > 0:
                rangos[i][0] = 0.0
                rangos[i][1] = attHere.getNumNominalValues() - 1
                # print(" attHere.getNumNominalValues() > 0,rangos[" + str(i) + "][0]==" + str(rangos[i][0]) + ",rangos[i][1]== " + str(rangos[i][1]))

            else:
                rangos[i][0] = attHere.getMinAttribute()
                rangos[i][1] = attHere.getMaxAttribute()
                # print(" attHere.getNumNominalValues() <= 0, rangos[" + str(i) + "][0]==" + str(rangos[i][0]) + ",rangos[i][1]== " + str(rangos[i][1]))

        rangos[self.get_nvars() - 1][0] = Attributes.getOutputAttribute(Attributes, 0).getMinAttribute()
        rangos[self.get_nvars() - 1][1] = Attributes.getOutputAttribute(Attributes, 0).getMaxAttribute()
        return rangos

    def get_granularity_zone_ranges(self, data_set_x_array):

        # print("self.get_nvars()" + str(self.get_nvars()))
        rangos = [[0.0 for y in range(2)] for x in range(self.get_nvars())]
        # print("rangos has two dimensions, first is self.get_nvars()==" + str(self.get_nvars()) + ",second is 2")
        nInputs = self.getn_inputs()
        for i in range(0, nInputs):
            # print("self.getn_inputs() is :" + str(nInputs) + " i = " + str(i))
            attHere = Attributes.getInputAttribute(Attributes, i)
            # print("attHere.getNumNominalValues()== " + str(attHere.getNumNominalValues()))
            if attHere.getNumNominalValues() > 0:
                rangos[i][0] = 0.0
                rangos[i][1] = attHere.getNumNominalValues() - 1
                # print(" attHere.getNumNominalValues() > 0,rangos[" + str(i) + "][0]==" + str(rangos[i][0]) + ",rangos[i][1]== " + str(rangos[i][1]))

            else:
                rangos[i][0] = attHere.get_min_granularity_attribute(data_set_x_array, i)
                rangos[i][1] = attHere.get_max_granularity_attribute(data_set_x_array, i)
                # print(" attHere.getNumNominalValues() <= 0, rangos[" + str(i) + "][0]==" + str(rangos[i][0]) + ",rangos[i][1]== " + str(rangos[i][1]))
        last_min_value = Attributes.getOutputAttribute(Attributes, 0).getMinAttribute()
        last_max_value = Attributes.getOutputAttribute(Attributes, 0).getMaxAttribute()
        # print("The last_min_value is " + str(last_min_value)+" The last_max_value is " + str(last_max_value))
        rangos[self.get_nvars() - 1][0] = last_min_value
        rangos[self.get_nvars() - 1][1] = last_max_value
        return rangos

    #    * It returns the attribute labels for the input features
    #    * @return String[] the attribute labels for the input features

    def get_names(self):
        names = ["" for x in range(self.ninputs)]
        for i in range(0, self.ninputs):
            names[i] = Attributes.getInputAttribute(Attributes, i).getName()
            print(" attributes' names["+str(i)+"]:" + names[i])
        return names

    #    * It returns the class labels
    #    * @return String[] the class labels

    def get_classes(self):
        clases = ["" for x in range(self.nclasses)]
        # print(" getClasses,self.nclasses: " + str(self.nclasses))
        for i in range(0, self.nclasses):
            # print(" getClasses method i is "+str(i))
            clases[i] = Attributes.getOutputAttribute(Attributes, 0).getNominalValue(i)
        return clases

    def is_nominal(self, index_i):
        return self.nominal_array[index_i]

    def is_integer(self, index_i):

        return self.integer_array[index_i]

    def get_frequent_class(self, class_value):
        print("class_value"+str(class_value))
        return self.frequent_class_array[class_value]

    """

     * It gets the number of input attributes of the data-set
     * @return int the number of input attributes of the data-set
    """

    def get_ninputs(self):
        return self.ninputs

    """
     * It returns the ratio of instances of the given class in the dataset
     *
     * @param clas the index of the class
     * @return the ratio of instances of the given class in the dataset
    """

    def frecuent_class(self, class_value):
        return self.frequent_class_array[class_value]

    def get_X(self):

        return np.array(self.x_array)

    def get_y(self, type_name='integer'):

        if type_name == "real":
            return np.array(self.output_real_array)
        elif type_name == "integer":
            return np.array(self.output_integer_array)
        else:
            return np.array(self.output_array)

    def copyHeader(self):

        p = ""
        # # print("copyHeader begin...., P is :" + p)
        p = "@relation " + Attributes.getRelationName(Attributes) + "\n"
        # # print(" after relation P is :" + p)
        p += Attributes.getInputAttributesHeader(Attributes)
        # # print(" after getInputAttributesHeader P is :" + p)
        p += Attributes.getOutputAttributesHeader(Attributes)
        # # print(" after getOutputAttributesHeader P is :" + p)
        p += Attributes.getInputHeader(Attributes) + "\n"
        # # print(" after getInputHeader P is :" + p)
        p += Attributes.getOutputHeader(Attributes) + "\n"
        # # print(" after getOutputHeader P is :" + p)
        p += "@data\n"

        # print("P is :" + p)
        return p
