# *********************************************************************
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
# *********************************************************************

# Attribute.java
#
# * <b> Attribute </b>
#
# * It contains an attribute representation. The class attributes are enough to
# * descrive completly an attribute: name, type, possible values, minimums and
# * maximums, etc. It offers a collection of functions to get all this information.
# *
# * @author Albert Orriols Puig
# * @version keel0.1
import numpy as np
class Attribute:
    #  '''
    # /////////////////////////////////////////////////////////////////////////////
    # //////////////// CONSTANTS OF THE ATTRIBUTE CLASS ///////////////////////////
    # /////////////////////////////////////////////////////////////////////////////
    #
    #
    #  * Label for NOMINAL values.
    # '''

    NOMINAL = 0

    # * Label for INTEGER values.

    INTEGER = 1
    # * Label for REAL VALUES

    REAL = 2

    #  Label to identify INPUT attributes

    INPUT = 1

    # Label to identify OUTPUT attributes

    OUTPUT = 2

    #  Label to identify attributes that hasn't been defined neither as input or output

    DIR_NOT_DEF = -1

    # '''
    # /////////////////////////////////////////////////////////////////////////////
    # /////////////// ATTRIBUTES OF THE ATTRIBUTE CLASS ///////////////////////////
    # /////////////////////////////////////////////////////////////////////////////
    #
    #
    #  * It indicates if the attribute  is an input (0), an output (1) or has not been
    #  * defined neither as input or output (-1)
    # '''

    __dirAttribute = None

    # '''
    #  * It keeps the type of the attribute. It can be one of the following values:
    #  * [Attribute.Nominal, Attribute.Integer, Attribute.Real]
    # '''

    __type = None

    # It stores the name of the attribute.

    __name = ""
    # '''
    #  * Vector where all the values that can take this nominal attribute are going
    #  * to be stored.
    #  '''

    __nominalValues = []
    # '''
    #  * Minimum value that can take a real attribute.
    #  '''

    __min = None
    # '''
    #  * Maximum value that can take a real attribute.
    #  '''

    __max = None
    # '''
    #  * Flag that indicates if it's the first time that an operation is made
    #  * with the current attribute.
    #  '''

    __firstTime = False
    # '''
    #  * It indicates if the bounds of the attribute has been fixed in its definition.
    #  '''

    _fixedBounds = False
    # '''
    #  * It counts the number of values that can take a nominal attribute
    #  '''

    __countValues = None

    # '''
    #  * It informs that a nominal value not compresed in train list values has been
    #  * read in test
    #  '''

    __newValuesInTest = False
    # '''
    #  * It keeps the new values in test
    #  '''

    __newValuesList = []
    # '''
    #  It keeps the frequency of each class value
    # '''

    #
    # '''
    # # Creates a list containing 5 lists, each of 8 items, all set to 0
    # w, h = 8, 5;
    # Matrix = [[0 for x in range(w)] for y in range(h)]
    # '''
    # w=0
    # h=0
    # __classFrequencies = [[0 for x in range(w)] for y in range(h)]
    __classFrequencies = []

    # '''
    #  * It stores the most used value in a nominal attribute
    #  '''

    # '''
    # strs = ["" for x in range(size)]
    # '''
    __mostUsedValue = []
    # '''
    #  * It stores the integer/real mean for this attribute
    #  '''

    __meanValue = []
    # '''
    #  * It keeps the number of updates per class
    #  '''

    __numStatUpdates = []

    # '''
    #  * It says if statistics has to be made
    #  '''

    __makeStatistics = None

    # '''
    # /////////////////////////////////////////////////////////////////////////////
    # ///////////////// METHODS OF THE ATTRIBUTE CLASS ////////////////////////////
    # /////////////////////////////////////////////////////////////////////////////
    #
    #
    #  * Attribute Constructor. It instances a new Attribute instance.
    #  '''

    def __init__(self):
        self.__type = -1
        self.__countValues = 0
        self.__dirAttribute = self.DIR_NOT_DEF
        self.__makeStatistics = False

    # end Attribute

    # '''
    #  * It sets the attribute type.
    #      * @param _type given attribute type
    #  '''

    def setType(self, type):
        if self.__type != -1:
            print("Type already fixed !!")
            exit(1)

        self.__type = type
        self.__firstTime = True
        # '''
        #     If type is nominal, a new vector has to be created to store the list of
        #     values that it can take.
        # '''
        if self.__type == self.NOMINAL:
            self.__nominalValues = []
            self.__newValuesList = []

        # //In all cases, the fixedBounds flag is set to false.
        self.__fixedBounds = False
        # end setType

    # '''
    #  * It does return the type of the attribute
    #  * @return an int that contains the type of the attribute.
    #  '''

    def getType(self):
        return self.__type

    # //end getType

    # '''
    #  * It sets the attribute name
    #  * @param _name is the name to be set.
    #  '''

    def setName(self, name):
        self.__name = name

    # end setName

    #
    # '''
    #  * It gets the attribute name
    #  * @return a String with the attribute name.
    #  '''

    def getName(self):
        return self.__name

    # end setName

    # '''
    #  * It sets the bound of the integer or real attribute.
    #  * @param _min is the minimum value that the attribute can take.
    #  * @param _max is the maximum value that the attribute can take.
    #  '''

    def setBounds(self, minBound, maxBound):
        if self.__type != self.REAL and self.__type != self.INTEGER:
            return
        else:
            self.__fixedBounds = True
            self.__min = minBound
            self.__max = maxBound

    # end setBounds

    # '''
    #  * It returns the variable fixedBounds.
    #  * @return a boolean that indicates if the bounds are fixed.
    #  '''

    def getFixedBounds(self):

        # print("self.__fixedBounds: " + str(self.__fixedBounds))
        return self.__fixedBounds

    # end getFixedBounds

    #
    # '''
    #  * It sets the fixedBounds value
    #  * @param fBounds is the value that has to be fixed to fixedBounds.
    #  '''

    def setFixedBounds(self, fBounds):
        self.__fixedBounds = fBounds

    # end setFixedBounds

    # '''
    #  * It does enlarge the attribute bounds
    #  * @param value is the value read from the BD file
    #  '''

    def enlargeBounds(self, value):
        if self.__type != self.REAL and self.__type != self.INTEGER:
            return

        if self.__firstTime:
            # //If it's the first attribute update and the bounds are not fixed in its
            # //specification, the min and max values are initialized.
            if not self.__fixedBounds:
                self.__min = value
                self.__max = value

            self.__firstTime = False
        # //valueMeans[instanceClass]+=value;
        self.__countValues += 1

        if self.__fixedBounds:
            return
        if value < self.__min:
            self.__min = value
        if value > self.__max:
            self.__max = value

    # end enlargeBounds

    # '''
    #  * It update an integer or real value read for an attribute in the test
    #  * set if it doesn't match with the bounds defined in the train set. In
    #  * this case, it replaces the value read for the nearliest bound (the
    #  * minimum or the maximum bound respectively)
    #  * @param value is the value read from the test file.
    #  * @return a double with the rectified value.
    #  '''

    def rectifyValueInBounds(self, value):
        if value < self.__min:
            return self.__min
        if value > self.__max:
            return self.__max
        return value

    # end rectifyValueInBounds

    # '''
    #  * It does check if the value passed as an argument is bounded by
    #  * the [min, max] interval.
    #      * @param val value to check.
    #  * @return a boolean that indicates if the value is bounded.
    #  '''

    def isInBounds(self, val):
        return self.__min <= val <= self.__max

    # end isInBounds

    # '''
    #  * It returns if the value passed is in the list of nominal values
    #  * @param val is the value to be checked.
    #  * @return a boolean indicating if the value is a possible nominal.
    #  '''

    def isNominalValue(self, val):
        return val in self.__nominalValues

    # end isNominalValue

    #
    # '''
    #  * It returns the minimum possible value in a integer or real attribute
    #  * @return a double with the minimum value
    #  '''

    def getMinAttribute(self):
        # print(" attribute the min value is " + str(self.__min))
        return self.__min

    def get_min_granularity_attribute(self, data_set_x_array, attribute_number):
        min_value = np.min(np.array(data_set_x_array)[:, attribute_number])
        return min_value
    # end minAttribute

    #
    # '''
    #  * It returns the maximum possible value in a integer or real attribute
    #  * @return a double with the maximum value
    #  '''
    def getMaxAttribute(self):
        print(" attribute the max value is " + str(self.__max))
        return self.__max

    def get_max_granularity_attribute(self, data_set_x_array, attribute_number):
        max_value = np.max(np.array(data_set_x_array)[:, attribute_number])
        return max_value

    # end maxAttribute

    # '''
    #  * This method add a new value to the list of possible values in a nominal
    #  * attribute.
    #  * @param value is the new value to be added.
    #  '''
    def addNominalValue(self, value):
        # print("addNominalValue begin......")
        if self.__type != self.NOMINAL:
            return
        if value not in self.__nominalValues:
            # print("self.__nominalValues ,add value:" + str(value) + ",to __nominalValues")
            self.__nominalValues.append(str(value))

    # end addNominalValue

    #
    # '''
    #  * It does return the value most frequent for the class
    #  * @param whichClass is the class which is wanted to know the most
    #  *        frequent value.
    #  * @return a String with the most used value.
    #  '''
    def getMostFrequentValue(self, whichClass):
        if not self.__makeStatistics or self.__type != self.NOMINAL or self.__mostUsedValue is None:
            return None
        if whichClass < 0 or whichClass >= len(self.__mostUsedValue):
            return None
        return self.__mostUsedValue[whichClass]

    # end getMostFrequentValue

    #
    # '''
    #  * Does return the mean value for that attribute.
    #  * @param whichClass is the integer value for the class
    #  * @return a double with the mean value.
    #  '''
    def getMeanValue(self, whichClass):
        if (not self.__makeStatistics or (
                self.__type != self.REAL and self.__type != self.INTEGER) or self.__meanValue is None):
            return 0
        if whichClass < 0 or whichClass >= len(self.__meanValue):
            return 0
        return self.__meanValue[whichClass]

    # end getMeanValue

    # '''
    #  * It does initializes the variables to make statistics
    #  * @param classNumber is the number of classes.
    #  '''
    def initStatisticsTwo(self, classNumber):
        self.__makeStatistics = True
        if self.__type == self.NOMINAL:
            # print("In initStatisticsTwo type is Nominal")
            # w, h = 8, 5;
            # Matrix = [[0 for x in range(w)] for y in range(h)]
            self.w = int(classNumber)
            # print("self.w = " + str(self.w))
            self.h = len(self.__nominalValues)
            # print("self.h = " + str(self.h))
            self.__classFrequencies = [[0 for y in range(self.h)] for x in range(classNumber)]
            self.__numStatUpdates = [0 for x in range(classNumber)]
            for i in range(0, classNumber):
                self.__numStatUpdates[i] = 0
                nominalValueLen = len(self.__nominalValues)
                for j in range(0, nominalValueLen):
                    self.__classFrequencies[i][j] = 0
                    # print("classFrequencies[i][j] " + "i" + str(i) + "j" + str(j))

        else:
            # print("In initStatisticsTwo type is not Nominal")
            # print("classNumber is :" + str(classNumber))
            self.__meanValue = [0.0 for x in range(classNumber)]
            self.__numStatUpdates = [0 for x in range(classNumber)]
            # print("before the loop in not nominal block")
            for i in range(0, classNumber):
                self.__meanValue[i] = 0.0
                self.__numStatUpdates[i] = 0
            # print("finished type is not Nominal.")

    # end initStatistics

    # '''
    #  * It does finish the statistics process.
    #  '''
    def finishStatistics(self):
        if not self.__makeStatistics:
            return
        if self.__type == self.NOMINAL:
            # print("In Attribute class the type is nominal, in finishStatistics method")
            mostUsedValue = ["" for x in range(len(self.__classFrequencies))]

            for i in range(0, len(mostUsedValue)):
                max_value = self.__classFrequencies[i][0]
                pos = 0
                for j in range(1, len(self.__classFrequencies[i])):
                    if self.__classFrequencies[i][j] > max_value:
                        max_value = self.__classFrequencies[i][j]
                        pos = j

                mostUsedValue[i] = str(self.__nominalValues[pos])

        else:
            # print("In Attribute class the type is not nominal, in finishStatistics method")
            for i in range(0, len(self.__meanValue)):
                # print("self.__meanValue[i]  is :" + str(self.__meanValue[i]))
                # print("float(self.__numStatUpdates[i])" + str(float(self.__numStatUpdates[i])))
                if self.__numStatUpdates[i] != 0:
                    self.__meanValue[i] = self.__meanValue[i] / float(self.__numStatUpdates[i])

    # end finishStatistics

    # '''
    #  * It does increment the frequency that a value of a class has been used.
    #  * It's called when a new value is read.
    #  * @param whichClass is the class which frequency has to be increased
    #  * @param value is the nominal value which frequency has to be increased.
    #  '''
    def increaseClassFrequency(self, whichClass, value):
        # print("increaseClassFrequency begin......")
        if self.__makeStatistics and self.__classFrequencies is not None and \
                self.__classFrequencies[whichClass] is not None:
            column_here = self.convertNominalValue(value)
            # print("self.__classFrequencies, row here is :" + whichClass + ",column_here is :" + str(column_here))
            self.__classFrequencies[whichClass][column_here] = self.__classFrequencies[whichClass][column_here] + 1
            self.__numStatUpdates[whichClass] = self.__numStatUpdates[whichClass] + 1

    # end increaseClassFrequency

    # '''
    #  * It adds the new value to the mean values vector
    #  * @param whichClass is the class where to add the new value
    #  * @param value is the value to be added.
    #  '''
    def addInMeanValue(self, whichClass, value):
        if self.__makeStatistics:
            self.__numStatUpdates[whichClass] += 1
            self.__meanValue[whichClass] += value

    # en addInMeanValue

    #
    # '''
    #  * Adds a new value for a nominal that has been read in the test file.
    #  * @param value is the new value to be added.
    #  * @return a boolean indicating if value didn't exist in the list.
    #  '''
    def addTestNominalValue(self, value):
        # print("addTestNominalValue begin......")
        if self.__type != self.NOMINAL:
            # print("The type is not NOMINAL, cannot add nominal value ")
            return False

        if value not in self.__nominalValues:
            # print("The type is not NOMINAL, cannot add nominal value ")
            self.__nominalValues.append(str(value))
            self.__newValuesList.append(str(value))
            self.__newValuesInTest = True
            return True

        return False
        # end addTestNominalValue

    # '''
    #  * It returns a vector with all new nominal values read in test.
    #  * @return a Vector with all new nominal values.
    #  '''
    def getNewValuesInTest(self):
        return self.__newValuesList

    # end newValuesList

    #
    # '''
    #  * It returns true if in test have appeared new values.
    #  * @return a boolean indicating if new values have been read in test.
    #  '''
    def areNewNominalValuesInTest(self):
        return self.__newValuesInTest

    # return areNewValuesInTest

    # '''
    #  * It returns the number of different values that can take a nominal attribute.
    #  * @return an int with the number of different values that can take a nominal
    #  *         attribute.
    #  '''
    def getNumNominalValues(self):
        if (self.__type != self.NOMINAL):
            return -1
        return len(self.__nominalValues)

    # end getNumNominalValues

    #
    # '''
    #  * Returns all the possible nominal values
    #  * @return a Vector with the possible values that the nominal can take
    #  '''
    def getNominalValuesList(self):
        return self.__nominalValues

    # end getNominalValuesList

    #
    # '''
    #  * It returns de ith value of that nominal attribute
    #  * @param pos indicate which attribute value is wanted.
    #  * @return a string with the value.
    #  '''
    def getNominalValue(self, pos):
        # # print("length of self.__nominalValues is: " + str(len(self.__nominalValues)))
        if self.__type != self.NOMINAL:
            return None
        # print("pos is in getNominalValue" + str(pos))
        return str(self.__nominalValues[pos])

    # end getNominalValue

    # '''
    #  * It converts a nominal value to a integer
    #  * @param value is the value that is wanted to be converted
    #  * @return an int with the converted value.
    #  '''
    def convertNominalValue(self, value):
        # print("convertNominalValue begin......")
        # for value_inside in self.__nominalValues:
        # print("value in self.__nominalValues is :" + str(value_inside))
        position_here = self.__nominalValues.index(value.strip())
        # print("for value pass :" + str(value) + ", position_here : " + str(position_here))
        return position_here

    # end convertNominalValue

    #
    # '''
    #  * It compares two attributes.
    #  * @param attr is the second attribute of the comparation.
    #  * @return a boolean that indicates if the attributes are equal.
    #  '''
    def equals(self, attr):
        if not self.__name == attr.name:
            return False
        if attr.type != self.__type:
            return False
        if self.__type == self.NOMINAL and (self.__nominalValues != attr.nominalValues):
            return False
        return True
        # end equals

    #
    #
    # '''
    #  * It sets if the attribute is an input or an output attribute
    #  * @param _dirAtt is the direction (input/output) of the attribute.
    #  '''
    def setDirectionAttribute(self, dirAtt):
        self.__dirAttribute = dirAtt

    # end setInputAttribute

    #
    # '''
    #  * It returns if the attribute is an input attribute
    #  * @return a int that indicates if it's an input or output attribute
    #  '''
    def getDirectionAttribute(self):
        return self.__dirAttribute

    # end getDirectionAttribute

    # '''
    #  * It does normalize a value.
    #  * @param val is the value to be normalized.
    #  * @return a double with the normalized value.
    #  '''
    def normalizeValue(self, val):
        if self.__type == self.NOMINAL:
            return val
        if self.__type == self.INTEGER:
            return val - self.__min
        if self.__type == self.REAL:
            return (val - min) / (self.__max - self.__min)
        return val

    # end normalizeValue

    # '''
    #  * It returns a String with the attribute information in keel format
    #  * @return an String with the attribute information.
    #  '''
    def toString(self):
        self.__typeNames = {"", "integer", "real"}
        aux = "@attribute " + str(self.__name)
        if self.__type == self.NOMINAL:

            aux += "{"
            ending = ","
            for i in range(0, len(self.__nominalValues)):
                if i == len(self.__nominalValues) - 1:
                    ending = ""
                aux += str(self.__nominalValues[i]) + ending

            aux += '}'
            # //System.out.println("Caso NOMINAL, aux->"+aux);
            # //System.out.println("name->" + name);
        elif self.__type == self.INTEGER:

            aux += " integer[" + str(int(self.__min))
            aux += "," + str(int(self.__max)) + "]"

        elif self.__type == self.REAL:
            aux += " real[" + str(float(self.__min))
            aux += "," + str(float(self.__max)) + "]"

        return aux

    # end toString
    #
    # '''
    #  * This method prints the attribute information.
    #  '''
    def printAttr(self):
        typesConv = {"Nominal", "Integer", "Real"}
        print(" Name: " + self.__name + ".")
        print(" Type: " + self.__type)

        for typeHere in typesConv:
            print(" Type: " + typeHere + ".")

        print(" Input/Output: ")
        if self.__dirAttribute == self.INPUT:

            print("INPUT")

        elif self.__dirAttribute == self.OUTPUT:
            print("OUTPUT")
        else:
            print("NOT DEFINED")

        print(" > Range: ")
        if self.__type == self.NOMINAL:
            print("{")
            for i in range(0, len(self.__nominalValues)):
                print(str(self.__nominalValues[i]) + "  ")
                print("}")
        elif self.__type == self.INTEGER:

            print("[" + int(self.__min) + "," + int(self.__max) + "]")

        else:
            print("[" + self.__min + "," + self.__max + "]")

        if (self.__type == self.NOMINAL) and (self.__mostUsedValue is not None):
            print("\n    > Most used value: ")
            for i in range(0, len(self.__mostUsedValue)):
                print("       > class " + str(i) + ":" + self.__mostUsedValue[i])
                print("  (" + self.__classFrequencies[i][self.convertNominalValue(self.__mostUsedValue[i])] + ").")

        elif self.__meanValue is not None:
            print("\n    > Mean used value: ")
            for i in range(0, len(self.__meanValue)):
                print("   > class " + str(i) + ": " + str(self.__meanValue[i]))

        print("\n")
        # end print
        # end of class Attribute
