# /***********************************************************************
#
# 	This file is part of KEEL-software, the Data Mining tool for regression,
# 	classification, clustering, pattern mining and so on.
#
# 	Copyright (C) 2004-2010
#
# 	F. Herrera (herrera@decsai.ugr.es)
#     L. S谩nchez (luciano@uniovi.es)
#     J. Alcal谩-Fdez (jalcala@decsai.ugr.es)
#     S. Garc铆a (sglopez@ujaen.es)
#     A. Fern谩ndez (alberto.fernandez@ujaen.es)
#     J. Luengo (julianlm@decsai.ugr.es)
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
# **********************************************************************/
#
# /*
#  * Created on 28 de enero de 2005, 9:07
#  */
#
# package keel.Dataset;
#
# /**
#  * <p>
#  * <b> ErrorInfo </b>
#  * </p>
#  * This class conatins the information about an error apperaed during the dataset
#  * read.
#  *
#  * @author Albert Orriols Puig
#  * @version keel0.1
#  *
#  */

from Help_Classes.Attribute import Attribute


class ErrorInfo:
    # /**
    #  * Definitions of possible ERRORS
    #  */
    BadNumberOfValues = 0
    OutputMissingValue = 1
    BadNumericValue = 2
    TrainNominalOutOfRange = 3
    TestNominalOutOfRange = 4
    TrainNumberOutOfRange = 5
    TestNumberOutOfRange = 6
    TypeAlreadyFixed = 7
    AttributeNotDefinedInTrain = 8
    InputTrainAttributeNotDefined = 9
    InputTestAttributeNotDefined = 10
    OutputTrainAttributeNotDefined = 11
    OutputTestAttributeNotDefined = 12
    InputsInTestNotEquals = 13
    OutputsInTestNotEquals = 14

    # /**
    #  * It stores the type of the error
    #  */
    typeOfError = 0

    # /**
    #  * It stores the instance number where the error has appeared.
    #  */
    instanceNum = 0

    # /**
    #  * It stores the file number where the error has appeared.
    #  */
    fileLineNum = 0

    # /**
    #  * It stores the attribute number where the error has appeared.
    #  */
    attributeNum = 0

    # /**
    #  * It keeps if the attribute is an input, output or non-defined attribute
    #  */
    attDirection = 0

    # /**
    #  * It stores if the error has been in the train dataset. Otherwise
    #  * it has been in the test dataset.
    #  */
    errorInTrain = None

    # /**
    #  * Message to be writen when showing the error
    #  */
    __message = ""

    # /**
    #  * Creates a new instance of ErrorInfo
    #  */
    def __init__(self):
        self.typeOfError = -1
        self.instanceNum = -1
        self.fileLineNum = -1
        self.attributeNum = -1
        self.attDirection = Attribute.DIR_NOT_DEF
        self.errorInTrain = False

    # end ErrorInfo

    #
    # /**
    #  * Creates a new instance with the parameters passed.
    #  */
    def set_Eight_Parameters(self, _type, _iNum, _lNum, _atNum, _atDir, _train, _msg):
        self.typeOfError = _type
        self.instanceNum = _iNum
        self.fileLineNum = _lNum
        self.attributeNum = _atNum
        self.attDirection = _atDir
        self.errorInTrain = _train
        self.message = _msg
        # end ErrorInfo

    #
    # /**
    #  * It creates a new Error info with the message passed
    #  * @param msg is the error message
    #  */
    def __init__(self, msg):
        self.message = msg
        self.typeOfError = -1
        self.instanceNum = -1
        self.fileLineNum = -1
        self.attributeNum = -1
        self.attDirection = Attribute.DIR_NOT_DEF
        self.errorInTrain = False

    # end ErrorInfo

    # /**
    #  * It does print an understable message about the error
    #  */
    def printErrorInfo(self):
        dir = ["Output", "Input"]
        if self.typeOfError == self.BadNumberOfValues:

            print(
                "BadNumberOfValuesException >> [line: " + self.fileLineNum + ", instance: " + self.instanceNum + ", Train DB: " + self.errorInTrain + "]")
            print(self.message)

        elif self.typeOfError == self.OutputMissingValue:
            print(
                "OutputMissingValueException >> [line: " + self.fileLineNum + ", instance: " + self.instanceNum + ", attributeNum: " + self.attributeNum + ", INPUT/OUTPUT: " +
                dir[Attribute.OUTPUT - self.attDirection] + " Train DB: " + self.errorInTrain + "]")
            print(self.message)

        elif self.typeOfError == self.BadNumericValue:
            print(
                "BadNumericValueException >> [line: " + self.fileLineNum + ", instance: " + self.instanceNum + ", attributeNum: " + self.attributeNum + ", INPUT/OUTPUT: " +
                dir[Attribute.OUTPUT - self.attDirection] + " Train DB: " + self.errorInTrain + "]")
            print(self.message)

        elif self.typeOfError == self.TrainNominalOutOfRange:
            print(
                "TrainNominalOutOfRangeException >> [line: " + self.fileLineNum + ", instance: " + self.instanceNum + ", attributeNum: " + self.attributeNum + ", INPUT/OUTPUT: " +
                dir[Attribute.OUTPUT - self.attDirection] + " Train DB: " + self.errorInTrain + "]")
            print(self.message)

        elif self.typeOfError == self.TestNominalOutOfRange:
            print(
                "TestNominalOutOfRangeException >> [line: " + self.fileLineNum + ", instance: " + self.instanceNum + ", attributeNum: " + self.attributeNum + ", INPUT/OUTPUT: " +
                dir[Attribute.OUTPUT - self.attDirection] + " Train DB: " + self.errorInTrain + "]")
            print(self.message)

        elif (self.typeOfError == self.TrainNumberOutOfRange):
            print(
                "TrainNumberOutOfRangeException >> [line: " + self.fileLineNum + ", instance: " + self.instanceNum + ", attributeNum: " + self.attributeNum + ", INPUT/OUTPUT: " +
                dir[Attribute.OUTPUT - self.attDirection] + " Train DB: " + self.errorInTrain + "]")
            print(self.message)

        elif self.typeOfError == self.TestNumberOutOfRange:
            print(
                "TestNumberOutOfRangeException >> [line: " + self.fileLineNum + ", instance: " + self.instanceNum + ", attributeNum: " + self.attributeNum + ", INPUT/OUTPUT: " +
                dir[Attribute.OUTPUT - self.attDirection] + " Train DB: " + self.errorInTrain + "]")
            print(self.message)

        elif self.typeOfError == self.TypeAlreadyFixed:
            print(
                "TypeAlreadyFixedException >> [line: " + self.fileLineNum + ", instance: " + self.instanceNum + ", attributeNum: " + self.attributeNum + ", INPUT/OUTPUT: " +
                dir[Attribute.OUTPUT - self.attDirection] + " Train DB: " + self.errorInTrain + "]");
            print(self.message)

        elif self.typeOfError == self.AttributeNotDefinedInTrain:

            print(
                "AttributeNotDefinedInTrainException >> [line: " + self.fileLineNum + ", attributeNum: " + self.attributeNum + ", Train DB: " + self.errorInTrain + "]")
            print(self.message)
        elif self.typeOfError == self.InputTrainAttributeNotDefined:

            print(
                "InputTrainAttributeNotDefinedException >> [line: " + self.fileLineNum + ", Train DB: " + self.errorInTrain + "]")
            print(self.message)
        elif self.typeOfError == self.InputTestAttributeNotDefined:

            print(
                "InputTestAttributeNotDefinedException >> [line: " + self.fileLineNum + ", Train DB: " + self.errorInTrain + "]")
            print(self.message)
        elif self.typeOfError == self.OutputTrainAttributeNotDefined:

            print(
                "OutputTrainAttributeNotDefinedException >> [line: " + self.fileLineNum + ", Train DB: " + self.errorInTrain + "]")
            print(self.message)
        elif self.typeOfError == self.OutputTestAttributeNotDefined:

            print(
                "OutputTestAttributeNotDefinedException >> [line: " + self.fileLineNum + ", Train DB: " + self.errorInTrain + "]")
            print(self.message)
        elif self.typeOfError == self.InputsInTestNotEquals:

            print("InputsInTestNotEqualsException >> [Train DB: " + self.errorInTrain + "]")
            print(self.message)
        elif self.typeOfError == self.OutputsInTestNotEquals:

            print("OutputsInTestNotEqualsException >> [Train DB: " + self.errorInTrain + "]")
            print(self.message)

# end print

#
# end of Class ErrorInfo
