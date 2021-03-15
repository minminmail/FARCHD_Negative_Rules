# '''
#     This class contains a complete set of instances. Its public methods are:
#     numInstances
#     Returns the number of instances of the Instance Set.
#     getInstance
#     Returns a concrete instance contained in the Instance Set.
#     getInstances
#     Returns an array with all the instances of the Instance Set.
# '''
# /***********************************************************************
#
# This file is part of KEEL-software, the Data Mining tool for regression,
# classification, clustering, pattern mining and so on.
#
# Copyright (C) 2004-2010
#
# F. Herrera (herrera@decsai.ugr.es)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see http://www.gnu.org/licenses/

# **********************************************************************/

#  * <p>
#  * <b> InstanceSet </b>
#  * </p>
#  *
#  * The instance set class mantains a pool of instances read from the keel
#  * formated data file. It provides a set of methods that permit to get
#  * each instance, get the whole set of instances, get the number of instances,
#  * etc.
#  *
#  * @author Albert Orriols Puig
#  * @version keel0.1
#  * @see Instance
#  * @see Attributes
#
import os

from Help_Classes.FormatErrorKeeper import FormatErrorKeeper
from Help_Classes.InstanceParser import InstanceParser
from Help_Classes.Attribute import Attribute
from Help_Classes.Attributes import Attributes
from Help_Classes.InstanceAttributes import InstanceAttributes
from Help_Classes.Instance import Instance
from Help_Classes.ErrorInfo import ErrorInfo
from pathlib import Path


class InstanceSet:
    # /////////////////////////////////////////////////////////////////////////////
    # //////////////// ATTRIBUTES OF THE INSTANCESET CLASS ////////////////////////
    # /////////////////////////////////////////////////////////////////////////////

    # Attribute where all the instances of the DB are stored.

    instanceSet = []

    # String where the header of the file is stored.

    header = ""

    # String where only the attributes definition header is stored
    attHeader = ""
    # '''
    #  * Object that collects all the errors happened while reading the test and
    #  * train datasets.
    # '''
    errorLogger = FormatErrorKeeper()

    # This object contains the attributes definitions

    attributes = InstanceAttributes()
    # '''
    #  * It indicates if the attributes has not be stored as non-static, permiting
    #  * the load of different datasets
    # '''
    storeAttributesAsNonStatic = None

    # It indicates that the output attribute has been infered as the last one

    outputInfered = None

    # /////////////////////////////////////////////////////////////////////////////
    # ///////////////// METHODS OF THE INSTANCESET CLASS //////////////////////////
    # /////////////////////////////////////////////////////////////////////////////

    # It instances a new instance of InstanceSet
    # data_folder = PureWindowsPath('C:/phd_experiments/threeAlgorithmsComparizasion/threeAlgorithmsComparizasion/ecoli')
    data_folder = None
    file_to_open = None
    data_lines = None

    # added by rui
    data_rows = None
    attributes_insance =None

    def __init__(self):
        # print("In __init__ method in InstanceSet.")
        self.storeAttributesAsNonStatic = False
        self.attributes = None
        self.attributes_insance  = Attributes()

    def InstanceSetWithNonSAtrr(self, nonStaticAttributes):
        self.storeAttributesAsNonStatic = nonStaticAttributes
        # if ( storeAttributesAsNonStatic ) Attributes.clearAll();
        self.attributes = None

    def InstanceSetWithIns(self, ins):
        self.instanceSet = ins.instanSet.copy()

        self.header = str(ins.header)
        self.attHeader = str(ins.attHeader)
        self.attributes = str(ins.attributes)
        self.storeAttributesAsNonStatic = ins.storeAttributesAsNonStatic

    # end InstanceSet

    # * InstanceSet
    # *
    # * This constructor permit define if the attribute's definition need to be
    # * stored as non-static (nonStaticAttributes = true). Otherwise, if
    # * nonStaticAttributes = false, using this constructor is equivalent to use
    # * the constructor by default.

    # * Creates a new InstanceSet with the header and Instances from the passed object
    # * It performs a deep (new allocated) copy.
    # * @param is Original InstanceSe

    # * setAttributesAsNonStatic
    # *
    # * It stores the static-defined attributes in the class Attributes as
    # * non static in the object attributes. After this it does not remove the
    # * static-definition of the Attributes; this is in that way to permit to
    # * call this functions for differents datasets from the same problem, such
    # * as, a train dataset and the correspondent test dataset.
    # */

    def setAttributesAsNonStatic(self):
        self.attributes = InstanceAttributes()
        self.attributes.copyStaticAttributes()

        self.storeAttributesAsNonStatic = True

    # end setAttributesAsNonStatic

    # /**
    #  * getAttributeDefinitions
    #  *
    #  * It does return the definition of the attibutes contained in the dataset.
    #  * 
    #  * @return InstanceAttributes contains the attribute's definitions.

    def getAttributeDefinitions(self):
        return self.attributes

    # end InstanceAttributes

    # * This method reads all the information in a DB and load it to memory.
    # * @param fileName is the database file name.
    # * @param isTrain is a flag that indicate if the database is for a train or for a test.
    # * @throws DatasetException if there is any semantical error in the input file.
    # * @throws HeaderFormatException if there is any lexical or sintactical error in the
    # * header of the input file

    def read_set(self, fileName, isTrain,file_path):
        print("Before try in readSet of InstanceSet, file_path is :" + str(file_path) + ".")
        print("Opening the file in readSet of InstanceSet: " + str(fileName) + ".")
        try:
            # Parsing the header of the DB.
            errorLogger = FormatErrorKeeper()
            self.file_to_open = Path.cwd() / file_path/fileName


            # Declaring an instance parser
            print("In readSet,file_to_open is:" + str(self.file_to_open))
            # to do The exception in init InstanceParserof InstanceParse is: can only concatenate str (not "WindowsPath") to str
            instance_parser = InstanceParser(self.file_to_open, isTrain)
            # Reading information in the header, i.e., @relation, @attribute, @inputs and @outputs
            # print("In readSet finished read file " + str(self.file_to_open))
            self.parseHeader(instance_parser, isTrain)
            # print(" The number of output attributes is: " + str(Attributes.getOutputNumAttributes(Attributes)))
            # The attributes statistics are init if we are in train mode.
            print("In readSet, isTrain is " + str(isTrain))
            if isTrain and Attributes.getOutputNumAttributes(Attributes) == 1:
                print("Begin Attributes.initStatistics......")
                Attributes.initStatistics(Attributes)
            # A temporal vector is used to store the instances read.

            print("Reading the data")
            tempSet = []
            print("begin instance_parser.getLines()...... ")

            new_data_lines = []
            print("*********  There are : " + str(len(self.data_lines)) + "In original Data lines ********* ")

            for line in self.data_lines:
                print(" The line is :"+ line)
                if ("@relation" not in line) and ("@attribute" not in line) and ("@inputs" not in line) and (
                        "@outputs" not in line) and ("@data" not in line):
                    new_data_lines.append(line)
            print("*********  There are : " + str(len(new_data_lines)) + " In new Data lines ********* ")

            for line in new_data_lines:
                if new_data_lines is not None:
                    #print("Data line: " + str(line))
                    newInstance = Instance()
                    #print("how many data already in the instanceSet: " + str(len(tempSet)))
                    newInstance.setThreeParameters(line, isTrain, len(tempSet))
                    tempSet.append(newInstance)

                # The vector of instances is converted to an array of instances.
            sizeInstance = len(tempSet)
            print(" Number of instances read: " + str(sizeInstance))
            self.instanceSet = []

            for i in range(0, sizeInstance):
                self.instanceSet.append(tempSet[i])
            print("After converting all instances")

            if self.errorLogger.getNumErrors() > 0:
                errorNumber = len(errorLogger.getAllErrors())
                # print("There has been " + str(errorNumber) + "errors in the Dataset format.")
                for k in range(0, errorLogger.getNumErrors()):
                    errorLogger.getError(k).printErrorInfo()

            # print("There has been " + errorLogger.getAllErrors().size() + " errors in the Dataset format",
            #           errorLogger.getAllErrors());
            # print("Finishing the statistics: (isTrain)" + str(isTrain) + ", (# out attributes)" + str(Attributes.getOutputNumAttributes(Attributes)))
            # # If being on a train dataset, the statistics are finished
            if isTrain and Attributes.getOutputNumAttributes(Attributes) == 1:
                Attributes.finishStatistics(Attributes)
            # # close the stream
            instance_parser.close()
            # print("File LOADED CORRECTLY!!")
        except Exception as e:
            print("Unexpected error in readSet of InstanceSet class :" + str(e))
        # end of InstanceSet constructor.

        # * It reads the information in the header of the file.
        # * It reads relation's name, attributes' names, and inputs and outputs.
        # *
        # * @param parser is the parser of the data set
        # * @param isTrain is a boolean indicating if this is a train set (and so
        # * parameters information must be read) or a test set (parameters information
        # * has not to be read).

        # read set from data row array for granularity
    def read_set_from_data_row_array(self, data_raw_array, isTrain):
        # print("Before try in read_set_from_data_row_array of InstanceSet")
        try:
            # Parsing the header of the DB.
            errorLogger = FormatErrorKeeper()
            # Declaring an instance parser

            # to do The exception in init InstanceParserof InstanceParse is: can only concatenate str (not "WindowsPath") to str
            instance_parser = InstanceParser.init_for_granularity_parser(data_raw_array, isTrain)
            # Reading information in the header, i.e., @relation, @attribute, @inputs and @outputs
            # print("data_raw_array size" + str(len(data_raw_array)))
            self.parse_header_from_data_row_array(instance_parser, isTrain)
            # print(" The number of output attributes is: " + str(Attributes.getOutputNumAttributes(Attributes)))
            # The attributes statistics are init if we are in train mode.
            # print("In readSet, isTrain is " + str(isTrain))
            if isTrain and Attributes.getOutputNumAttributes(Attributes) == 1:
                # print("Begin Attributes.initStatistics......")
                Attributes.initStatistics(Attributes)
            # A temporal vector is used to store the instances read.

            # print("Reading the data in read_set_from_data_row_array")
            tempSet = []
            # print("begin instance_parser.getLines()...... ")
            data_raw_array = self.data_rows
            new_data_rows = []
            number_of_rows= len(data_raw_array)
            # print("*********  There are : " + str(number_of_rows) + "In original Data rows ********* ")

            # print("*********  There are : " + str(len(new_data_rows)) + " In new Data rows ********* ")
            for i in range(0, number_of_rows):
                if len(new_data_rows) != 0:
                    # print("Data row: " + str(data_raw_array[i]))
                    newInstance = Instance()
                    # print("how many data already in the instanceSet: " + str(len(tempSet)))
                    newInstance.set_three_parameters_for_granularity_rules(data_raw_array[i], isTrain, len(tempSet))
                    tempSet.append(newInstance)

                # The vector of instances is converted to an array of instances.
            sizeInstance = len(tempSet)
            # print(" Number of instances read: " + str(sizeInstance))
            self.instanceSet = []

            for i in range(0, sizeInstance):
                self.instanceSet.append(tempSet[i])
            # print("After converting all instances")
            # System.out.println("The error logger has any error: "+errorLogger.getNumErrors());
            if self.errorLogger.getNumErrors() > 0:
                errorNumber = len(errorLogger.getAllErrors())
                # print("There has been " + str(errorNumber) + "errors in the Dataset format.")
                for k in range(0, errorLogger.getNumErrors()):
                    errorLogger.getError(k).printErrorInfo()

            # print("There has been " + errorLogger.getAllErrors().size() + " errors in the Dataset format",
            #           errorLogger.getAllErrors());
            # print("Finishing the statistics: (isTrain)" + str(isTrain) + ", (# out attributes)" + str(Attributes.getOutputNumAttributes(Attributes)))
            # # If being on a train dataset, the statistics are finished
            if isTrain and Attributes.getOutputNumAttributes(Attributes) == 1:
                Attributes.finishStatistics(Attributes)
            # # close the stream
            instance_parser.close()
            # print("File LOADED CORRECTLY!!")
        except Exception as e:
            print("Unexpected error in readSet of InstanceSet class :" + str(e))
        # end of InstanceSet constructor.


    def parseHeader(self, parser, isTrain):
        # 1. Declaration of variables
        inputAttrNames = []
        outputAttrNames = []
        inputsDef = False
        outputsDef = False
        self.header = ""
        attCount = 0
        lineCount = 0
        self.attHeader = None

        print("Begin to call the InstanceParser.getLines(),parser.getLines(), in InstanceSet.")
        lines = parser.getLines()
        self.data_lines = lines

        print("************************Before for line in lines *************************")
        for line in lines:
            line = str(line).strip()
            print("In parseHeader method of InstanceSet, the line is:" + line)
            if line == "@data".lower():

                break
            else:
                # print("  Line read: " + line + ".")
                lineCount = lineCount + 1
                if "@relation" in line:

                    if isTrain:
                        relationName = str(line.replace("@relation", "")).strip()
                        # print("set Relation name :" + str(relationName))
                        self.attributes_insance.setRelationName(self.attributes_insance,relationName)
                elif "@attribute" in line:

                    if isTrain:
                        # print("Begin insertAttribute ......")
                        self.insertAttribute(line)
                        attCount = attCount + 1

                elif "@inputs" in line:

                    # print("@inputs in " + str(line))
                    self.attHeader = self.header
                    inputsDef = True

                    aux = line[8:]

                    if isTrain:
                        # print("Has @inputs, aux is :" + aux)
                        self.insertInputOutput(aux, lineCount, inputAttrNames, "inputs", isTrain)
                elif "@outputs" in line:

                    if self.attHeader is None:
                        self.attHeader = self.header
                    outputsDef = True
                    # print("Defining the output in line :" + line)
                    sub_line = line.split()  # To get the output attribute name
                    aux = sub_line[1]
                    if isTrain:
                        # print("Has @outputs, aux is :" + aux)
                        self.insertInputOutput(aux, lineCount, outputAttrNames, "outputs", isTrain)

                        # print("Size of the output is: " + str(len(outputAttrNames)))

                self.header += line + "\n"
        print("************************After for line in lines.************************")
        if self.attHeader is None:
            self.attHeader = self.header
        self.processInputsAndOutputs(isTrain, inputsDef, outputsDef, outputAttrNames, inputAttrNames)

    # end headerParse

    # added by rui for granularity rules
    def parse_header_from_data_row_array(self, parser, isTrain):
        # 1. Declaration of variables
        inputAttrNames = []
        outputAttrNames = []
        inputsDef = False
        outputsDef = False
        self.header = ""
        attCount = 0
        lineCount = 0
        self.attHeader = None

        # print("Begin to call the InstanceParser.getLines(),parser.getLines(), in InstanceSet.")
        self.data_rows = parser.get_rows()

    # end parse_header_from_data_row_array

    def insertAttribute(self, line):
        # print("Insert attribute begin :")
        indexL = 0
        indexR = 0
        type = ""

        # Treating string and declaring a string tokenizer
        if "{" in line:
            token_str = "{"

        elif "[" in line:
            token_str = "["

        token_withT = "\t" + token_str

        line = line.replace(token_str, token_withT)
        # print("token_double is:" + token_withT + ", line is :" + line)
        # System.out.println ("  > Processing line: "+  line );
        # st = line.split(" [{\t");

        st = line.split(
            "\t")  # first we need to split the attribute line into two part , attribute name and attribute values

        # Disregarding the first token. It is @attribute
        st[0] = st[0].replace("@attribute", "").strip()  # delete @attribute
        # print("st[0] is:" + st[0])

        first_part = st[0].split()

        at = Attribute()

        # print("Get type once get instance object, at.getType() = " + str(type_string))
        at.setName(first_part[0])
        print("att set name as first_part[0] is:" + first_part[0])
        # # print( "Attribute name: "+ at.getName() )

        # to get the class name values we need to split the second part of the attribute line, to get values of attribute

        # Next action depends on the type of attribute: continuous or nominal
        if len(st) == 1:  # Parsing a nominal attribute with no definition of values
            # print("Parsing nominal attribute without values: setType=0")
            # print("Get type =" + at.getType())
            at.setType(Attribute.NOMINAL)

        elif "{" in line:  # this because  it is the class values line
            # print("Parsing nominal attribute with values: " + line)
            # print("Get type =" + at.getType())
            # print("Before setType = 0")
            at.setType(Attribute.NOMINAL)
            # print("after setType= 0")
            at.setFixedBounds(True)

            indexL = line.index("{") + 1
            # print("indexL: " + indexL )
            indexR = line.index("}")
            # print("indexR: " + str(indexR))
            print("indexL : " + str(indexL) + "indexR : " + str(indexR))
            # print( "The Nominal values are: " + line[indexL: indexR]);
            lineSub = line[indexL: indexR]
            # print("The lineSub : " + lineSub)
            st2 = lineSub.split(",")

            for nominalStr in st2:
                at.addNominalValue(nominalStr.strip())

        else:  # Parsing an integer or real

            attType = first_part[1].lower()
            # print("attribute Name : " + str(first_part[0]) + ", attribute type = " + str(attType))

            # System.out.println ("    > Parsing "+ type + " attributes");

            if attType == "integer":
                at.setType(Attribute.INTEGER)
                # print("set integer type")
            if attType == "real":
                at.setType(Attribute.REAL)
                # print("set real type")
            indexL = line.index("[")
            indexR = line.index("]")

            # print("indexL is: " + str(indexL) + " indexR: " + str(indexR))

            if indexL != -1 and indexR != - 1:
                # System.out.println ( "      > The real values are: " + line.substring( indexL+1, indexR) );
                lineSub = line[indexL + 1: indexR]
                # print("lineSub: " + lineSub)
                st2 = lineSub.split(",")

                # print("st2[0].strip() :" + st2[0])
                # print("st2[1].strip() :" + st2[1])
                minBound = float(st2[0].strip())
                maxBound = float(st2[1].strip())
                # print("Before at.setBounds(minBound, maxBound): ( " + str(minBound) + " , " + str(maxBound) + " )")
                at.setBounds(minBound, maxBound)

        # print("Before add attribute :::: ")
        Attributes.addAttribute(Attributes, at)
        # print("insertAttribute is finished :::: ")

    # end insertAttribute

    def insertInputOutput(self, line, lineCount, collection, type, isTrain):

        # print(" processing insertInputOutput: " + line)

        # Declaring StringTokenizer
        st = line.split(",")

        for attName in st:
            attName = str(attName.strip())
            # print("attrName: " + attName)
            attrItem = Attributes.getAttributeByName(Attributes, attName)
            attributes = Attributes.getAttributes(Attributes)
            # for att in attributes:
                # print("att name is :" + str(att.getName()))
            # print("numbers of items that attributes:"+str(len(attributes)))
            if attrItem is None:
                # print("Attributes.getAttribute == None")
                # If this attribute has not been declared, generate error
                er = ErrorInfo(ErrorInfo.InputTestAttributeNotDefined, 0, lineCount, 0, 0, isTrain,
                               ("The attribute " + attName + " defined in @" + type +
                                " in test, it has not been defined in @inputs in its train dataset. It will be ignored"))
                InstanceSet.errorLogger.setError(er)

            else:
                # for itemCollection in collection:
                    # print("Item in collection is " + itemCollection)
                # print("Attributes.getAttribute != None")
                # print("   > " + str(type) + " attribute considered: " + attName)
                if attName not in collection:
                    # print("attName:" + attName + " is not in collection")
                    collection.append(attName)

    # end insertInputOutput

    def processInputsAndOutputs(self, isTrain, inputsDef, outputsDef, outputAttrNames, inputAttrNames):
        # After parsing the header, the inputs and the outputs are prepared.
        print("Processing inputs and outputs")
        self.outputInfered = False  # set default value
        if isTrain:
            print("isTrain == True")
            if not inputsDef and not outputsDef:
                # print("is neither inputAtt no outputAtt")
                posHere = Attributes.getNumAttributes(self) - 1

                outputAttrNames.append(Attributes.getAttributeByPos(self, posHere).getName())
                inputAttrNames = Attributes.getAttributesExcept(Attributes, outputAttrNames)
                self.outputInfered = True
            elif not inputsDef and outputsDef:
                # print("inputsDef == False and outputsDef == True")
                inputAttrNames = Attributes.getAttributesExcept(Attributes, outputAttrNames)
            elif inputsDef and not outputsDef:
                # print("inputsDef == True and outputsDef == False")
                outputAttrNames = Attributes.getAttributesExcept(Attributes, inputAttrNames)
                self.outputInfered = True
            print("setOutputInputAttributes begin: ")
            Attributes.setOutputInputAttributes(Attributes, inputAttrNames, outputAttrNames)

    # end of processInputsAndOutputs

    # '''
    #  * Test if the output attribute has been infered.
    #  * @return True if the output attribute has been infered. False if not.
    #  '''

    def isOutputInfered(self):
        return self.outputInfered

    # '''
    #  * It returns the number of instances.
    #  * @return an int with the number of instances.
    # '''

    def getNumInstances(self):
        if self.instanceSet is not None:
            instanceNumber = len(self.instanceSet)
            print("instanceSet is not None, instanceNumber = " + str(instanceNumber))
            return instanceNumber
        else:
            print("instanceSet is  None !!!")
            return 0
        # end numInstances

    # '''
    #  * Gets the instance located at the cursor position.
    #  * @return the instance located at the cursor position.
    # '''

    def getInstance(self, whichInstance):
        if whichInstance < 0 or whichInstance >= len(self.instanceSet):
            return None
        return self.instanceSet[whichInstance]

    # end getInstance

    #  * It returns all the instances of the class.
    #  * @return Instance[] with all the instances of the class.

    def getInstances(self):
        return self.instanceSet

    # //end getInstances
    # '''
    # '''
    #  * Returns the value of an integer or a real input attribute of an instance
    #  * in the instanceSet.
    #  * @param whichInst is the position of the instance.
    #  * @param whichAttr is the position of the input attribute.
    #  * @return a String with the numeric value.
    #  * @throws ArrayIndexOutOfBoundsException If the index is out of the instance
    #  * set size.
    # '''

    def getInputNumericValue(self, whichInst, whichAttr):
        # print("InstanceSet, getInputNumericValue begin...")
        instance_number = len(self.instanceSet)
        # print("whichInst = " + str(whichInst) + ", whichAttr =" + str(whichAttr))
        # print("len(self.instanceSet) = " + str(instance_number))

        if whichInst < 0 or whichInst >= instance_number:
            raise IndexError("You are trying to access to " + whichInst + " instance and there are only " + str(
                instance_number) + ".")
        instanceHere = self.instanceSet[whichInst]
        #print("instanceHere = " + str(instanceHere))
        numericValue = 0.0
        try:
            numericValue = instanceHere.getInputRealValues(whichAttr)
        except Exception as error:
            print("getInputRealValues has exception!! : " + str(error))

        return numericValue

    # end getInputNumericValue

    # '''
    #  * Returns the value of an integer or a real output attribute of an instance
    #  * in the instanceSet.
    #  * @param whichInst is the position of the instance.
    #  * @param whichAttr is the position of the output attribute.
    #  * @return a String with the numeric value.
    #  * @throws ArrayIndexOutOfBoundsException If the index is out of the instance
    #  * set size.
    # '''

    def getOutputNumericValue(self, whichInst, whichAttr):
        if whichInst < 0 or whichInst >= len(self.instanceSet):
            print(self.ArrayIndexOutOfBoundsException("You are trying to access to " + whichInst + "instance and there are only" + self.instanceSet.length + "."))
        return self.instanceSet[whichInst].getOutputRealValues(whichAttr)
        # end getOutputNumericValue

    #
    # '''
    #  * Returns the value of a nominal input attribute of an instance in the
    #  * instanceSet.
    #  * @param whichInst is the position of the instance.
    #  * @param whichAttr is the position of the input attribute.
    #  * @return a String with the nominal value.
    #  * @throws ArrayIndexOutOfBoundsException If the index is out of the instance
    #  * set size.
    # '''

    def getInputNominalValue(self, whichInst, whichAttr):
        if whichInst < 0 or whichInst >= len(self.instanceSet):
            print(self.ArrayIndexOutOfBoundsException("You are trying to access to " + whichInst + " instance and there are only " + str(
                    len(self.instanceSet)) + "."))
        return self.instanceSet[whichInst].getOutputNominalValues(whichAttr)
        # end getInputNominalValue

    #
    # '''
    #  * Returns the value of a nominal output attribute of an instance in the
    #  * instanceSet.
    #  * @param whichInst is the position of the instance.
    #  * @param whichAttr is the position of the output attribute.
    #  * @return a String with the nominal value.
    #  * @throws ArrayIndexOutOfBoundsException If the index is out of the instance
    #  * set size.
    # '''

    def getOutputNominalValue(self, whichInst, whichAttr):
        if whichInst < 0 or whichInst >= len(self.instanceSet):
            print("You are trying to access to " + whichInst + " instance and there are only " + str(
                len(self.instanceSet)) + ".")
        return self.instanceSet[whichInst].getOutputNominalValues(whichAttr)
        # end getOutputNumericValue

    # '''
    #  * It does remove the instance i from the instanceSet.
    #  * @param instNum is the instance removed from the instanceSet.
    # '''

    def removeInstance(self, instNum):
        if instNum < 0 or instNum >= len(self.instanceSet):
            return
        aux = [Instance() for x in range(len(self.instanceSet) - 1)]
        add = 0
        for i in range(0, len(self.instanceSet)):
            if instNum == i:
                add = 1
            else:
                aux[i - add] = self.instanceSet[i]

        # Copying the auxiliar to the instanceSet variable
        self.instanceSet = aux
        aux = None  # avoiding memory leaks (not necessary in this case)

    # end removeInstance

    # '''
    #  * It does remove an attribute. To remove an attribute, the train and the
    #  * test sets have to be passed to mantain the coherence of the system.
    #  * Otherwise, only the attribute of the train set would be removed, leaving
    #  * inconsistent the instances of the test set, because of having one extra
    #  * attribute inexistent anymore.
    #  *
    #  * @param tSet is the test set.
    #  * @param inputAtt is a boolean that is true when the attribute that is
    #  * wanted to be removed is an input attribute.
    #  * @param whichAtt is a integer that indicate the position of the attriubte
    #  * to be deleted.
    #  * @return a boolean indicating if the attribute has been deleted
    # '''

    def removeAttribute(self, tSet, inputAtt, whichAtt):
        attToDel = None
        # Getting a reference to the attribute to del
        if inputAtt:
            if self.storeAttributesAsNonStatic and self.attributes is not None:
                attToDel = self.attributes.getInputAttribute(whichAtt)
            else:
                attToDel = Attributes.getInputAttribute(whichAtt)

        else:
            if self.storeAttributesAsNonStatic and self.attributes is not None:
                attToDel = self.attributes.getOutputAttribute(whichAtt)
            else:
                attToDel = Attributes.getOutputAttribute(whichAtt)

        if self.storeAttributesAsNonStatic and self.attributes is not None:
            print("Removing the attribute")
        if (not self.attributes.removeAttribute(inputAtt, whichAtt) or (
                tSet is not None and not tSet.attributes.removeAttribute(inputAtt, whichAtt))):
            return False
        else:
            if not Attributes.removeAttribute(inputAtt, whichAtt):
                return False
        for i in range(0, len(self.instanceSet)):
            if self.storeAttributesAsNonStatic and self.attributes is not None:
                self.instanceSet[i].removeAttribute(self.attributes, attToDel, inputAtt, whichAtt)
            else:
                self.instanceSet[i].removeAttribute(attToDel, inputAtt, whichAtt)

        if tSet is not None:
            for i in range(0, tSet.instanceSet.length):

                if self.storeAttributesAsNonStatic  and self.attributes is not None:
                    tSet.instanceSet[i].removeAttribute(self.attributes, attToDel, inputAtt, whichAtt)
            else:
                tSet.instanceSet[i].removeAttribute(attToDel, inputAtt, whichAtt)
            return True

    # end removeAttribute

    # '''
    #  * It returns the header.
    #  * @return a String with the header of the file.
    # '''

    def getHeader(self):
        return self.header

    # end getHeader

    def setHeader(self, copia):
        self.header = str(copia)

    # end getHeader

    def getAttHeader(self):
        return self.attHeader

    # end getHeader

    def setAttHeader(self, copia):
        self.attHeader = str(copia)

    # end getHeader

    # '''
    #  * It does return a new header (not necessary the same header as the
    #  * input file one). It only includes the valid attributes, those ones
    #  * defined in @inputs and @outputs (or taken as that role following the
    #  * keel format specification).
    #  * @return a String with the new header
    # '''

    def getNewHeader(self):
        line = ""
        attrs = []

        # Getting the relation name and the attributes
        if self.storeAttributesAsNonStatic and self.attributes is not None:
            line = "@relation " + self.attributes.getRelationName() + "\n"
            attrs = self.attributes.getInputAttributes(Attributes)
        else:
            line = "@relation " + Attributes.getRelationName() + "\n"
            attrs = Attributes.getInputAttributes(Attributes)

        for i in range(0, attrs.length):
            line += attrs[i].toString() + "\n"
            # Gettin all the outputs attributes
        if self.storeAttributesAsNonStatic and self.attributes is not None:
            attrs = self.attributes.getOutputAttributes()
            line += attrs[0].toString() + "\n"
            # Getting @inputs and @outputs
            line += self.attributes.getInputHeader() + "\n"
            line += self.attributes.getOutputHeader() + "\n"

        else:
            attrs = Attributes.getOutputAttributes()
            line += str(attrs[0]) + "\n"

        # Getting @inputs and @outputs
        line += Attributes.getInputHeader() + "\n"
        line += Attributes.getOutputHeader() + "\n"

        return line

    # end getNewHeader

    # '''
    #  * It does return the original header definiton but
    #  * without @input and @output in there
    # '''

    def getOriginalHeaderWithoutInOut(self):

        line = ""
        attrs = []

        # Getting the relation name and the attributes
        if self.storeAttributesAsNonStatic and self.attributes is not None:
            line = "@relation " + self.attributes.getRelationName() + "\n"
            attrs = self.attributes.getAttributes()

        else:
            line = "@relation " + Attributes.getRelationName() + "\n"
            attrs = Attributes.getAttributes()

        for i in range(0, len(attrs)):
            line = line + str(attrs[i]) + "\n"
        return line
        # end getOriginalHeaderWithoutInOut;

    # '''
    #  * It prints the dataset to the specified PrintWriter
    #  * @param out is the PrintWriter where to print
    # '''

    def printOut(self, out):
        for i in range(0, len(self.instanceSet)):
            print("> Instance " + i + ":")
        if self.storeAttributesAsNonStatic  and self.attributes is not None:
            self.instanceSet[i].printOut(self.attributes, out)
        else:
            self.instanceSet[i].printOut(out)

    # end print
    #
    # '''
    #  * It prints the dataset to the specified PrintWriter.
    #  * The order of the attributes is the same as in the
    #  * original file
    #  * @param out is the PrintWriter where to print
    #  * @param printInOut indicates if the @inputs (1), @outputs(2),
    #  * both of them (3) or any (0) has to be printed
    # '''

    def printAsOriginal(self, out, int):
        # Printing the header as the original one
        # print(self.header)

        if self.storeAttributesAsNonStatic and self.attributes is not None:
            if self.printInOut == 1 or self.printInOut == 3:
                print(self.attributes.getInputHeader())

        if self.printInOut == 2 or self.printInOut == 3:

            print(self.attributes.getOutputHeader())

        else:
            if self.printInOut == 1 or self.printInOut == 3:
                out.println(Attributes.getInputHeader())
            if self.printInOut == 2 or self.printInOut == 3:
                out.println(Attributes.getOutputHeader())

        print("@data")
        for i in range(0, len(self.instanceSet)):
            print()
            if self.storeAttributesAsNonStatic and self.attributes is not None:
                self.instanceSet[i].printAsOriginal(self.attributes, out)
        else:
            self.instanceSet[i].printAsOriginal(self.attributes,out)

    # end printAsOriginal

    def printInsSet(self):
        print("------------- ATTRIBUTES --------------")
        if self.storeAttributesAsNonStatic and self.attributes is not None:
            self.attributes.printAttributes()

        else:
            Attributes.printAttributes()

        print("-------------- INSTANCES --------------")
        for i in range(0, self.instanceSet.length):
            print("\n> Instance " + str(i) + ":")

            if self.storeAttributesAsNonStatic and self.attributes is not None:
                self.instanceSet[i].printInsSet(self.attributes)
        else:
            self.instanceSet[i].printInsSet()

    # end print

    # Remove all instances from this InstanceSet

    def clearInstances(self):
        self.instanceSet = None

    # '''
    #    * It adds the passed instance at the end of the present InstanceSet
    #    * @param inst the instance to be added
    # '''

    def addInstance(self, inst):
        i = 0
        nVector = []
        if self.instanceSet is not None:
            nVector = [Instance() for x in range(len(self.instanceSet) + 1)]
            for i in range(0, len(self.instanceSet)):
                nVector[i] = self.instanceSet[i]

        else:
            nVector = Instance[1]

        nVector[i] = inst
        self.instanceSet = nVector

    # '''
    #    * Clear the non-Static attributes. The static class Attributes is not modified.
    # '''

    def clearNonStaticAttributes(self):
        self.attributes = None

    # '''
    #    * Appends the given attribute to the non-static list of the current InstanceSet
    #    * @param at The Attribute to be Appended
    # '''

    def addAttribute(self, att):
        if self.attributes is None:
            self.attributes = InstanceAttributes()
        self.attributes.addAttribute(att)

    # end of InstanceSet Class.
