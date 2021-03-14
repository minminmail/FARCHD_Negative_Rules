# /***********************************************************************
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
# **********************************************************************/


# /**
#  * <p>
#  * <b>Instance</b>
#  * </p>
#  *
#  * This class keeps all the information of an instance. It stores nominal,
#  * integer and real values read from the file (in KEEL format). Also, it
#  * provides a set of methods to get information about the instance.
#  *
#  * @author Albert Orriols Puig
#  * @version keel0.1
#  */
import decimal

from Help_Classes.Attribute import Attribute
from Help_Classes.Attributes import Attributes
from Help_Classes.InstanceParser import InstanceParser
from Help_Classes.ErrorInfo import ErrorInfo
from decimal import *

decimal.setcontext(decimal.Context(prec=34))

import math


class Instance:
    # /////////////////////////////////////////////////////////////////////////////
    # ////////////////ATTRIBUTES OF THE INSTANCE CLASS ///////////////////////////
    # /////////////////////////////////////////////////////////////////////////////
    #
    # 	/**
    # 	 * It is a vector of vectors of size 'number of attributes' where all the nominal
    # 	 * values will of the attributes be stored. In nominalValues[0] the input values
    # 	 * are stored, and int nominalValues[1] the output values are stored.

    __nominalValues = []
    # /**
    #  * It is a vector of vector of size 'number of attributes' where all the nominal
    #  * values will of the attributes be stored, but transformed to a integer value.
    #  */
    __intNominalValues = []
    # * The vector realValues is a vector of vectors of size 'number of attributes'
    # * where all the integer and real attributes values will be stored. In realValues[0]
    # * all the input attribute values will be stored, while in realValues[1], the
    # * outputs will be stored.
    __realValues = []
    # * It is a vector of vectors of 'number of attributes' size that stores whichs
    # * attributes are missing for the inputs and the outputs.
    __missingValues = []

    # * Indicates if the instance belongs to a train BD
    isTrain = None

    # Indicates the number of input attributes per instance.
    __numInputAttributes = 0

    # Indicates the number of output attributes per instance.
    __numOutputAttributes = 0

    # * Indicates the number of undefined attributes (that are neither
    # * inputs or outputs
    __numUndefinedAttributes = 0

    # It indicates if the instance has any missing value
    __anyMissingValue = []

    # The next attriubtes define the position in the arrays where each attribute is stored
    # * Input attributes location
    ATT_INPUT = 0
    # Output attributes location
    ATT_OUTPUT = 1
    # Non-defined direction attributes location
    ATT_NONDEF = 2

    # /////////////////////////////////////////////////////////////////////////////
    # /////////////////// METHODS OF THE INSTANCE CLASS ///////////////////////////
    # /////////////////////////////////////////////////////////////////////////////
    #
    # /**
    #  * It parses a new attribute line.
    #  * @param def is the line to be parsed.
    #  * @param _isTrain is a flag that indicates if the BD is for a train run.
    #  * @param instanceNum is the number of the current instance. It's used to
    #  * write error message with the maximum amount of information.
    #  */
    def __init__(self):
        # print("__init__ of Instance begin......")
        self.__nominalValues = []


    def setThreeParameters(self, defStr, _isTrain, instanceNum):
        # print("setThreeParameters begin...... ")
        currentClass = -1
        # System.out.println ("Reading data: "+def);
        print("In setThreeParameters,defStr is : " + defStr)
        print("In setThreeParameters, instanceNum is : " + str(instanceNum))
        st = defStr.split(",")  # Separator: "," and " "
        print("inside setThreeParameters st length is :" + str(len(st)))
        self.initClassAttributes()
        self.isTrain = _isTrain

        count = 0
        inAttCount = 0
        outAttCount = 0
        indefCount = 0
        inputOutput = 0
        curCount = 0
        for att in st:
            att = att.strip()
            # Looking if the attribute is an input, an output or it's undefined

            curAt = Attributes.getAttributeByPos(Attributes, count)
            #print("inside setThreeParameters curAtis :" + str(curAt))
            directionAttr = curAt.getDirectionAttribute()
            # print("inside setThreeParameters directionAttr :" + str(directionAttr))
            if directionAttr == Attribute.INPUT:
                # print("directionAttr==Attribute.INPUT")
                inputOutput = Instance.ATT_INPUT
                curCount = inAttCount
                # print("curCount is : " + str(curCount))
                inAttCount = inAttCount + 1
                # print("inAttCount is : " + str(inAttCount))

            elif directionAttr == Attribute.OUTPUT:
                # print("directionAttr==Attribute.OUTPUT")
                inputOutput = Instance.ATT_OUTPUT
                if curAt.getType() == Attribute.NOMINAL:
                    # print("curAt.getType() == Attribute.NOMINAL")
                    currentClass = curAt.convertNominalValue(att)

                    # System.out.println ( " The position of the current class "+ att +" is: "+ currentClass );
                curCount = outAttCount
                outAttCount = outAttCount + 1

            else:
                # print("Attribute not defined neither as input or output")
                # Attribute not defined neither as input or output. So, it is not read.
                inputOutput = Instance.ATT_NONDEF
                curCount = indefCount
                indefCount = indefCount + 1

            # The attribute is defined. So, its value is processed, and the attributes definitions
            # are checked to detect inconsistencies or to redefine undefined traits.
            # print("Before processReadValue......")
            self.processReadValue(curAt, defStr, att, inputOutput, count, curCount, instanceNum)
            # print("After processReadValue......")
            # Finally, the counter of read attributes is updated.
            count = count + 1
            # end of the while

        # Checking if the instance doesn't have the same number of attributes than defined.
        if count != Attributes.getNumAttributes(Attributes):
            # print("count != Attributes.getNumAttributes(Attributes)......")
            er = ErrorInfo(ErrorInfo.BadNumberOfValues, instanceNum, InstanceParser.lineCounter, 0, 0, self.isTrain, (
                    "Instance " + defStr + " has a different number of attributes than defined\n   > Number of attributes defined: " + Attributes.getNumAttributes() + "   > Number of attributes read:    " + count))
            # InstanceSet.errorLogger.setError(er)

        # Compute the statistics
        if self.isTrain:
            # print("self.isTrain==True......")
            atts = Attributes.getInputAttributes(Attributes)
            length = int(len(atts))
            for i in range(0, length):
                if not self.__missingValues[Instance.ATT_INPUT][i]:
                    if (atts[i].getType() == Attribute.NOMINAL) and (Attributes.getOutputNumAttributes(Attributes) == 1):
                        atts[i].increaseClassFrequency(currentClass, self.__nominalValues[Instance.ATT_INPUT][i])
                    elif ((atts[i].getType() == Attribute.INTEGER or atts[i].getType() == Attribute.REAL) and
                          not self.__missingValues[Instance.ATT_INPUT][i]):
                        atts[i].addInMeanValue(currentClass, self.__realValues[Instance.ATT_INPUT][i])

        # print("setThreeParameters finished...... ")

    # end Instance

   # added by rui for granularity rules
    def set_three_parameters_for_granularity_rules(self, data_row, _isTrain, instanceNum):
        # print("setThreeParameters begin...... ")
        currentClass = -1
        st = data_row

        self.initClassAttributes()
        self.isTrain = _isTrain

        count = 0
        inAttCount = 0
        outAttCount = 0
        indefCount = 0
        inputOutput = 0
        curCount = 0
        for att in st:
            att = att.strip()
            # Looking if the attribute is an input, an output or it's undefined

            curAt = Attributes.getAttributeByPos(Attributes, count)
            # print("inside setThreeParameters curAtis :" + str(curAt))
            directionAttr = curAt.getDirectionAttribute()
            # print("inside setThreeParameters directionAttr :" + str(directionAttr))
            if directionAttr == Attribute.INPUT:
                # print("directionAttr==Attribute.INPUT")
                inputOutput = Instance.ATT_INPUT
                curCount = inAttCount
                # print("curCount is : " + str(curCount))
                inAttCount = inAttCount + 1
                # print("inAttCount is : " + str(inAttCount))

            elif directionAttr == Attribute.OUTPUT:
                # print("directionAttr==Attribute.OUTPUT")
                inputOutput = Instance.ATT_OUTPUT
                if curAt.getType() == Attribute.NOMINAL:
                    print("curAt.getType() == Attribute.NOMINAL")
                    currentClass = curAt.convertNominalValue(att)

                    # System.out.println ( " The position of the current class "+ att +" is: "+ currentClass );
                curCount = outAttCount
                outAttCount = outAttCount + 1

            else:
                # print("Attribute not defined neither as input or output")
                # Attribute not defined neither as input or output. So, it is not read.
                inputOutput = Instance.ATT_NONDEF
                curCount = indefCount
                indefCount = indefCount + 1

            # The attribute is defined. So, its value is processed, and the attributes definitions
            # are checked to detect inconsistencies or to redefine undefined traits.
            # print("Before processReadValue......")
            defStr=""
            for feature in  data_row.feature_values:
                defStr=defStr+feature+","
            defStr = defStr[:-1]
            self.processReadValue(curAt, defStr, att, inputOutput, count, curCount, instanceNum)
            # print("After processReadValue......")
            # Finally, the counter of read attributes is updated.
            count = count + 1
            # end of the while

        # Checking if the instance doesn't have the same number of attributes than defined.
        if count != Attributes.getNumAttributes(Attributes):
            print("count != Attributes.getNumAttributes(Attributes)......")
            er = ErrorInfo(ErrorInfo.BadNumberOfValues, instanceNum, InstanceParser.lineCounter, 0, 0, self.isTrain, (
                    "Instance " + defStr + " has a different number of attributes than defined\n   > Number of attributes defined: " + Attributes.getNumAttributes() + "   > Number of attributes read:    " + count))
            # InstanceSet.errorLogger.setError(er)

        # Compute the statistics
        if self.isTrain:
            # print("self.isTrain==True......")
            atts = Attributes.getInputAttributes(Attributes)
            length = int(len(atts))
            for i in range(0, length):
                if not self.__missingValues[Instance.ATT_INPUT][i]:
                    if (atts[i].getType() == Attribute.NOMINAL) and (Attributes.getOutputNumAttributes(Attributes) == 1):
                        atts[i].increaseClassFrequency(currentClass, self.__nominalValues[Instance.ATT_INPUT][i])
                    elif ((atts[i].getType() == Attribute.INTEGER or atts[i].getType() == Attribute.REAL) and
                          not self.__missingValues[Instance.ATT_INPUT][i]):
                        atts[i].addInMeanValue(currentClass, self.__realValues[Instance.ATT_INPUT][i])

        # print("setThreeParameters finished...... ")

    # end Instance

    # * Creates a deep copy of the Instance
    # * @param inst Original Instance to be copied

    def setOneParameter(self, inst):
        print("setOneParameter......")
        self.isTrain = inst.isTrain
        self.__numInputAttributes = inst.__numInputAttributes
        self.__numOutputAttributes = inst.__numOutputAttributes
        self.__numUndefinedAttributes = inst.__numUndefinedAttributes

        self.__anyMissingValue = list.copyOf(inst.anyMissingValue, len(inst.anyMissingValue))
        # str[inst.nominalValues.length][];
        self.__nominalValues = ["" for x in range(len(inst.nominalValues))]
        for i in range(0, len(self.__nominalValues)):
            self.__nominalValues[i] = list.copyOf(inst.nominalValues[i], len(inst.__nominalValues[i]))
            # int[inst.intNominalValues.length][];
        self.__intNominalValues = len(inst.intNominalValues)
        for i in range(0, len(self.__nominalValues)):
            self.__intNominalValues[i] = list.copyOf(inst.__intNominalValues[i], len(inst.__intNominalValues[i]))

            # float[inst.realValues.length][];
        self.__realValues = float[len(inst.__realValues)]
        for i in range(0, len(self.__realValues)):
            self.__realValues[i] = list.copyOf(inst.realValues[i], len(inst.realValues[i]))

        # bool[inst.missingValues.length][];
        self.__missingValues = bool[len(inst.__missingValues)]
        for i in range(0, len(self.__missingValues)):
            self.__missingValues[i] = list.copyOf(inst.__missingValues[i], len(inst.__missingValues[i]))

    # * Creates an instance from a set of given values. It is supposed that the values
    # * correspond to the current Attributes static definition or InstanceAttributes
    # * non-static definition (it depends on the InstanceSet to which this new instance
    # * will belong). If ats is null, we will use the Attributes static definiton. If not
    # * the ats definition instead.
    # * @param values A double array with the values (either real or nominals' index). Missing values are stored as Double.NaN
    # * @param ats The definition of the attributes (optional, if null we use Attributes definition).

    def setTwoParameters(self, values, instanceAttrs):
        print("setTwoParameters......")
        curAt = Attribute()
        allat = []

        inOut = None
        inHere = None
        outHere = None
        undef = None

        # initialise structures
        self.anyMissingValue = [False for x in range(0, 3)]
        self.anyMissingValue[0] = False
        self.anyMissingValue[1] = False
        self.anyMissingValue[2] = False
        if instanceAttrs is None:
            self.__numInputAttributes = Attributes.getInputNumAttributes()
            self.__numOutputAttributes = Attributes.getOutputNumAttributes(Attributes)
            self.__numUndefinedAttributes = Attributes.getNumAttributes() - (
                    self.__numInputAttributes + self.__numOutputAttributes)
        else:
            self.__numInputAttributes = instanceAttrs.getInputNumAttributes()
            self.__numOutputAttributes = instanceAttrs.getOutputNumAttributes(Attributes)
            self.__numUndefinedAttributes = instanceAttrs.getNumAttributes() - (
                    self.__numInputAttributes + self.__numOutputAttributes)

        self.__intNominalValues = []
        self.__nominalValues = []
        self.__realValues = []
        self.__missingValues = []
        self.__nominalValues[0] = ["" for x in range(0, self.__numInputAttributes)]
        self.__nominalValues[1] = ["" for x in range(0, self.__numOutputAttributes)]
        self.__nominalValues[2] = ["" for x in range(0, self.__numUndefinedAttributes)]
        self.__intNominalValues[0] = [0 for x in range(0, self.__numInputAttributes)]
        self.__intNominalValues[1] = [0 for x in range(0, self.__numOutputAttributes)]
        self.__intNominalValues[2] = [0 for x in range(0, self.__numUndefinedAttributes)]
        self.__realValues[0] = [0.0 for x in range(0, self.__numInputAttributes)]
        self.__realValues[1] = [0.0 for x in range(0, self.__numOutputAttributes)]
        self.__realValues[2] = [0.0 for x in range(0, self.__numUndefinedAttributes)]
        self.__missingValues[0] = [0.0 for x in range(0, self.__numInputAttributes)]
        self.__missingValues[1] = [0.0 for x in range(0, self.__numOutputAttributes)]
        self.__missingValues[2] = [False for x in range(0, self.__numUndefinedAttributes)]

        for i in range(0, self.__numInputAttributes):
            self.__missingValues[0][i] = False
        for i in range(0, self.__numOutputAttributes):
            self.__missingValues[1][i] = False
        for i in range(0, self.__numUndefinedAttributes):
            self.__missingValues[2][i] = False

        # take the correct set of Attributes
        if instanceAttrs is not None:
            allat = instanceAttrs.getAttributes()
        else:
            allat = Attributes.getAttributes()

        # fill the data structures
        inHere = outHere = undef = 0
        for i in range(0, len(values)):
            curAt = allat[i]
            inOut = 2
            if curAt.getDirectionAttribute() == Attribute.INPUT:
                inOut = 0
            elif curAt.getDirectionAttribute() == Attribute.OUTPUT:
                inOut = 1

            # is it missing?
            if math.isnan(float(values[i])):
                if inOut == 0:
                    self.__missingValues[inOut][inHere] = True
                    self.anyMissingValue[inOut] = True
                    inHere += 1
                elif inOut == 1:
                    self.__missingValues[inOut][outHere] = True
                    self.anyMissingValue[inOut] = True
                    outHere += 1
                else:
                    self.__missingValues[inOut][undef] = True
                    self.anyMissingValue[inOut] = True
                    undef += 1

            elif not curAt.getType() == Attribute.NOMINAL:  # is not numerical
                if inOut == 0:
                    self.__realValues[inOut][inHere] = values[i]
                    inHere = inHere + 1
                elif inOut == 1:
                    self.__realValues[inOut][outHere] = values[i]
                    outHere = outHere + 1
                else:
                    self.__realValues[inOut][undef] = values[i]
                    undef = undef + 1

            else:  # is nominal

                if inOut == 0:
                    self.__intNominalValues[inOut][inHere] = int(values[i])
                    self.__realValues[inOut][inHere] = values[i]
                    self.__nominalValues[inOut][inHere] = curAt.getNominalValue(int(values[i]))
                    inHere += 1
                elif inOut == 1:
                    self.__intNominalValues[inOut][outHere] = int(values[i])
                    self.__realValues[inOut][outHere] = values[i]
                    self.__nominalValues[inOut][outHere] = curAt.getNominalValue(int(values[i]))
                    outHere = outHere + 1
                else:
                    self.__intNominalValues[inOut][undef] = int(values[i])
                    self.__realValues[inOut][undef] = values[i]
                    self.__nominalValues[inOut][undef] = curAt.getNominalValue(int(values[i]))
                    undef = undef + 1

    #
    # '''
    # /**
    #  * It processes the read value for an attribute, for example, convert class name into class category number
    #  * @param curAtt is the current attribute (the value read is from this attribute)
    #  * @param def is the whole String
    #  * @param inOut is an integer that indicates if the attribute is an input or an output attribute
    #  * @param count is a counter of attributes.
    #  * @param curCount is an attribute counter relative to the inputs or the output. So, it indicates
    #  * that the attribute is the ith attribute of the input or the output.
    #  * @param instanceNum is the number of the current instance. It's needed to write output messages
    #  * with the maximum possible amount of information.
    #  */

    def processReadValue(self, curAtt, defStr, att, inOut, count, curCount, instanceNum):
        # print("processReadValue begin......")
        # print("In processReadValue,count = " + str(count))
        # Checking if there is a missing value.
        # print(" In processReadValue, att = " + att)
        if att is None or att == "?":
            print("att==None or att==?......")
            Attributes.hasMissing = True
            self.__missingValues[inOut][curCount] = True
            self.__anyMissingValue[inOut] = True

            if inOut == 1:  # If the output is a missing value, an error is generated.
                error_info_1 = ErrorInfo(ErrorInfo.OutputMissingValue, instanceNum,
                                         InstanceParser.lineCounter, curCount, Attribute.OUTPUT,
                                         self.isTrain,
                                         ("Output attribute " + count + " of " + defStr + " with missing value."))
                # !!!! we need to fix the import problem , currently we only print th error
                # InstanceSet.errorLogger.setError(er)
                print(" !!!!!!!!! InstanceSet.errorLogger.setError: " + str(error_info_1))

        elif (Attributes.getAttributeByPos(Attributes,
                                           count).getType() == Attribute.INTEGER or Attributes.getAttributeByPos(
            Attributes, count).getType() == Attribute.REAL):
            # print("getType()==Attribute.INTEGER or Real ......")
            try:
                # print("inOut is:" + str(inOut) + ", curCount is: " + str(curCount))
                # print("The length of self.__realValues[0] column is:  " + str(len(self.__realValues[0])))
                self.__realValues[inOut][curCount] = float(att)
                # print("self.__realValues[" + str(inOut) + "][" + str(curCount) + "]=" + att)
            except  ValueError as valueError:
                error_info_2 = ErrorInfo(ErrorInfo.BadNumericValue, instanceNum,
                                         InstanceParser.lineCounter, curCount, Attribute.INPUT + inOut,
                                         self.isTrain,
                                         ("Attribute " + count + " of " + defStr + " is not an integer or real value."))
                print("There is an valueError in Instance init method:" + str(valueError))
                # InstanceSet.errorLogger.setError(valueError)
                print(" !!!!!!!!! InstanceSet.errorLogger.setError: " + str(error_info_2))
                print(" !!!!!!!!! InstanceSet.errorLogger.setError: " + str(valueError))

            # Checking if the new train value exceedes the bounds definition in train. The condition
            # also checks if the attribute is defined (is an input or an output).
            if self.isTrain and inOut != 2:
                # print("self.isTrain and inOut != 2......")
                if curAtt.getFixedBounds() and (not curAtt.isInBounds(self.__realValues[inOut][curCount])):
                    error_info_3 = ErrorInfo(ErrorInfo.TrainNumberOutOfRange, instanceNum, InstanceParser.lineCounter,
                                             curCount, Attribute.INPUT + inOut, self.isTrain,
                                             ("ERROR READING TRAIN FILE. Value " + str(self.__realValues[inOut][
                                                                                           curCount]) + " read for a numeric attribute that is not in the bounds fixed in the attribute '" + curAtt.getName() + "' definition."));
                    # InstanceSet.errorLogger.setError(er)
                    print(" !!!!!!!!! InstanceSet.errorLogger.setError: " + str(error_info_3))

                curAtt.enlargeBounds(self.__realValues[inOut][curCount])

            elif inOut != 2:  # In test mode
                # print("self.isTrain and inOut != 2......")
                self.__realValues[inOut][curCount] = curAtt.rectifyValueInBounds(self.__realValues[inOut][curCount]);

        elif Attributes.getAttributeByPos(Attributes, count).getType() == Attribute.NOMINAL:
            # print("getType()==Attribute.NOMINAL......")
            # print("inOut : " + str(inOut) + " , curCount: " + str(curCount))
            self.__nominalValues[inOut][curCount] = att
            # Testing special cases.
            if self.isTrain and inOut != 2:
                # print("self.isTrain and inOut!=2 begin......")

                rowLength = len(self.__nominalValues)
                # print("rowLength = " + str(rowLength))
                columnLength = len(self.__nominalValues[0])
                # print("inOut = " + str(inOut) + ",rowLength: " + str(rowLength) + "curCount:" + str(curCount) + ",columnLength: " + str(columnLength))
                nominalValue = self.__nominalValues[inOut][curCount]
                # print("nominalValue: " + str(nominalValue))
                if curAtt.getFixedBounds() and not curAtt.isNominalValue(nominalValue):
                    print("There are error_info_4!! ")
                    error_info_4 = ErrorInfo.set_Eight_Parameters(ErrorInfo, ErrorInfo.TrainNominalOutOfRange,
                                                                  instanceNum, InstanceParser.lineCounter, curCount,
                                                                  Attribute.INPUT + inOut, self.isTrain, (
                                                                          "ERROR READING TRAIN FILE. Value '" +
                                                                          self.__nominalValues[inOut][
                                                                              curCount] + "' read for a nominal attribute that is not in the possible list of values fixed in the attribute '" + curAtt.getName() + "' definition."));
                    # InstanceSet.errorLogger.setError(er)
                    print(" !!!!!!!!! InstanceSet.errorLogger.setError: " + str(error_info_4))

                curAtt.addNominalValue(self.__nominalValues[inOut][curCount])
                # print("self.isTrain and inOut!=2 finished......")
            elif inOut != 2:
                # print(" inOut!=2......")
                if curAtt.addTestNominalValue(self.__nominalValues[inOut][curCount]):
                    error_info_5 = ErrorInfo(ErrorInfo.TestNominalOutOfRange, instanceNum, InstanceParser.lineCounter,
                                             curCount, Attribute.INPUT + inOut, self.isTrain,
                                             ("ERROR READING TEST FILE. Value '" + self.__nominalValues[inOut][
                                                 curCount] + "' read for a nominal attribute that is not in the "
                                                             "possible list of values fixed in the attribute '" +
                                              curAtt.getName() + "' definition."));
                    # InstanceSet.errorLogger.setError(er);
                    print(" !!!!!!!!! InstanceSet.errorLogger.setError: " + str(error_info_5))

            if inOut != -2:
                # print(" inOut != -2......")
                self.__intNominalValues[inOut][curCount] = curAtt.convertNominalValue(
                    self.__nominalValues[inOut][curCount])
                self.__realValues[inOut][curCount] = self.__intNominalValues[inOut][curCount]

                # end processReadValue
        # print("processReadValue finished......")

    # It reserves all the memory necessary for this instance

    def initClassAttributes(self):
        # print("initClassAttributes begin......")
        # print("Length of self.__anyMissingValue  is : " + str(len(self.__anyMissingValue)))
        self.__anyMissingValue = [False for x in range(3)]
        self.__anyMissingValue[0] = False
        self.__anyMissingValue[1] = False
        self.__anyMissingValue[2] = False
        self.__numInputAttributes = Attributes.getInputNumAttributes(Attributes)
        self.__numOutputAttributes = Attributes.getOutputNumAttributes(Attributes)
        self.__numUndefinedAttributes = Attributes.getNumAttributes(Attributes) - (
                self.__numInputAttributes + self.__numOutputAttributes)
        self.__intNominalValues = [0 for x in range(3)]
        self.__nominalValues = ["" for x in range(3)]
        self.__realValues = [0.0 for x in range(3)]
        self.__missingValues = [False for x in range(3)]
        # print("Length of self.__numInputAttributes  is : " + str(self.__numInputAttributes))
        # print("Length of self.__numOutputAttributes  is : " + str(self.__numOutputAttributes))
        # print("Length of self.__numUndefinedAttributes  is : " + str(self.__numUndefinedAttributes))

        # print("Length of self.__nominalValues  is : " + str(len(self.__nominalValues)))
        self.__nominalValues[0] = ["" for x in range(self.__numInputAttributes)]
        self.__nominalValues[1] = ["" for x in range(self.__numOutputAttributes)]
        self.__nominalValues[2] = ["" for x in range(self.__numUndefinedAttributes)]
        # print("Length of self.__intNominalValues  is : " + str(len(self.__intNominalValues)))
        self.__intNominalValues[0] = [0 for x in range(self.__numInputAttributes)]
        self.__intNominalValues[1] = [0 for x in range(self.__numOutputAttributes)]
        self.__intNominalValues[2] = [0 for x in range(self.__numUndefinedAttributes)]
        # print("Length of self.__realValues  is : " + str(len(self.__realValues)))
        self.__realValues[0] = [0.0 for x in range(self.__numInputAttributes)]
        # print("Length of self.__realValues[0],__numInputAttributes,  is : " + str(self.__numInputAttributes))
        self.__realValues[1] = [0.0 for x in range(self.__numOutputAttributes)]
        # print("Length of self.__realValues[1] ,__numOutputAttributes, is : " + str(self.__numOutputAttributes))
        self.__realValues[2] = [0.0 for x in range(self.__numUndefinedAttributes)]
        # print("Length of self.__realValues[2] ,__numUndefinedAttributes, is : " + str(self.__numUndefinedAttributes))
        # print("Length of self.__missingValues  is : " + str(len(self.__missingValues)))
        self.__missingValues[0] = [0.0 for x in range(self.__numInputAttributes)]
        self.__missingValues[1] = [0.0 for x in range(self.__numOutputAttributes)]
        self.__missingValues[2] = [0.0 for x in range(self.__numUndefinedAttributes)]

        for i in range(0, self.__numInputAttributes):
            self.__missingValues[0][i] = False
        for i in range(0, self.__numOutputAttributes):
            self.__missingValues[1][i] = False

        for i in range(0, self.__numUndefinedAttributes):
            self.__missingValues[2][i] = False

        # print("initClassAttributes finished......")

    # end initClassAttributes

    # * It prints the instance to the specified PrintWriter.
    # * @param out is the PrintWriter where to print.
    """
    def printInstance(self, outHere):
        outHere.print("    > Inputs: ")
        for i in range(0, self.__numInputAttributes):
            attrType = Attributes.getInputAttribute(i).getType()

            if attrType == Attribute.NOMINAL:
                outHere.print(self.__nominalValues[Instance.ATT_INPUT][i])

            elif attrType == Attribute.INTEGER:
                outHere.print(self.__realValues[Instance.ATT_INPUT][i])

            elif attrType == Attribute.REAL:
                outHere.print(self.__realValues[Instance.ATT_INPUT][i])

            outHere.print("\n    > Outputs: ")
        for i in range(0, self.__numOutputAttributes):
            attrType = Attributes.getOutputAttribute(i).getType()

            if attrType == Attribute.NOMINAL:
                outHere.print(self.__nominalValues[Instance.ATT_OUTPUT][i])

            elif attrType == Attribute.INTEGER:
                outHere.print(self.__realValues[Instance.ATT_OUTPUT][i])

            elif attrType == Attribute.REAL:
                outHere.print(self.__realValues[Instance.ATT_OUTPUT][i])

            outHere.print("\n    > Undefined: ")
        for i in range(0, self.__numUndefinedAttributes):
            attrType = Attributes.getOutputAttribute(i).getType()

            if attrType == Attribute.NOMINAL:
                outHere.print(self.__nominalValues[Instance.ATT_OUTPUT][i])

            if attrType == Attribute.INTEGER:
                outHere.print(self.__realValues[Instance.ATT_OUTPUT][i])

            if attrType == Attribute.REAL:
                outHere.print(self.__realValues[Instance.ATT_OUTPUT][i])

    # end print
      """

    # * It prints the instance to the specified PrintWriter.
    # * The attribtes order is the same as the one in the
    # * original file.
    # * @param out is the PrintWriter where to print.

    def printAsOriginal(self, out):
        inCount = 0
        outCount = 0
        undefCount = 0
        count = 0
        numAttributes = Attributes.getNumAttributes()
        for count in range(0, numAttributes):
            at = Attributes.getAttributeByPos(count)
            directionAttr = at.getDirectionAttribute()

            if directionAttr == Attribute.INPUT:
                self.printAttribute(out, Instance.ATT_INPUT, inCount, at.getType())
                inCount += 1

            elif directionAttr == Attribute.OUTPUT:
                self.printAttribute(out, Instance.ATT_OUTPUT, outCount, at.getType())
                outCount += 1

            elif directionAttr == Attribute.DIR_NOT_DEF:
                self.printAttribute(out, Instance.ATT_NONDEF, undefCount, at.getType())
                undefCount += 1

            if count + 1 < numAttributes:
                out.printInsAttr(",")

    # end printAsOriginal

    # Does print an attribute to a PrintWriter

    def printAttribute(self, out, inOut, ct, type):

        if self.__missingValues[inOut][ct]:
            out.printInsAttr("<null>")

        else:

            if type == Attribute.NOMINAL:
                out.printInsAttr(self.__nominalValues[inOut][ct])

            elif type == Attribute.INTEGER:
                out.printInsAttr(int(self.__realValues[inOut][ct]))

            elif type == Attribute.REAL:
                out.printInsAttr(self.__realValues[inOut][ct])

    # end printAttribute

    # /**
    #  * It does print the instance information
    #  */

    def printFunction(self):
        print("  > Inputs (" + self.__numInputAttributes + "): ")
        for i in range(0, self.__numInputAttributes):
            if self.__missingValues[Instance.ATT_INPUT][i]:
                print("?")

            else:
                inputAttrType = Attributes.getInputAttribute(i).getType()
                if inputAttrType == Attribute.NOMINAL:
                    print(self.__nominalValues[Instance.ATT_INPUT][i])

                if inputAttrType == Attribute.INTEGER:
                    print(int(self.__realValues[Instance.ATT_INPUT][i]))

                if inputAttrType == Attribute.REAL:
                    print(self.__realValues[Instance.ATT_INPUT][i])

            print("  ")

        print("  > Outputs (" + self.__numOutputAttributes + "): ")
        for i in range(0, self.__numOutputAttributes):
            if self.__missingValues[Instance.ATT_OUTPUT][i]:
                print("?")

            else:
                outputAttr = Attributes.getOutputAttribute(i).getType()
                if outputAttr == Attribute.NOMINAL:
                    print(self.__nominalValues[Instance.ATT_OUTPUT][i])

                elif outputAttr == Attribute.INTEGER:
                    print(int(self.__realValues[Instance.ATT_OUTPUT][i]))

                elif outputAttr == Attribute.REAL:
                    print(self.__realValues[Instance.ATT_OUTPUT][i]);

            print("  ")
        print("  > Undefined (" + self.__numUndefinedAttributes + "): ")
        for i in range(0, self.__numUndefinedAttributes):
            if self.__missingValues[Instance.ATT_NONDEF][i]:
                print("?")

            else:
                undefinedAttrType = Attributes.getUndefinedAttribute(i).getType()

                if undefinedAttrType == Attribute.NOMINAL:
                    print(self.__nominalValues[Instance.ATT_NONDEF][i])
                elif undefinedAttrType == Attribute.INTEGER:

                    print(self.__realValues[Instance.ATT_NONDEF][i])

                elif undefinedAttrType == Attribute.REAL:
                    print(self.__realValues[Instance.ATT_NONDEF][i])

        print("  ")

    # end print

    # /////////////////////////////////////////////////////////////////////////////
    # ////////////////////////GET AND SET METHODS ////////////////////////////////
    # /////////////////////////////////////////////////////////////////////////////
    #
    # /////////////////////////////////////////////////////////////////////////////
    # //	Functions to get all the input attributes, or all the output attributes //
    # /////////////////////////////////////////////////////////////////////////////

    # * Get Input Real Values
    # * @return a double[] of size equal to the number of input attributes
    # * with the values of real attributes. Positions of the vector that doesn't
    # * correspond to a real attribute has no rellevant data.

    def getInputRealValues(self):
        print("getInputRealValues begin......")

        try:

            print(" len(self.__realValues)= " + str(len(self.__realValues)))
            if self.__realValues[0] is not None:
                return self.__realValues[0]
            else:
                return 0
        except Exception as error:
            print("inside getInputRealValues method, it has error:" + str(error))

    # end getInputRealAttributes

    # * Get Input Nominal Values
    # * @return a string[] of size equal to the number of input attributes
    # * with the values of nominal attributes. Positions of the vector that
    # * doesn't correspond to a nominal attribute has no rellevant data.

    def getInputNominalValues(self):
        return self.__nominalValues[0]

    # end getInputNominalValues

    # * Get Input Missing Values
    # * @return a boolean[] of size equal to the number of input attributes.
    # * A true value in the position i of a vector indicates that the ith
    # * input value is not known.

    def getInputMissingValues(self):
        print("getInputMissingValues begin......")
        return self.__missingValues[0]

    # end getINputMissingValues

    # * Get Output Real Values
    # * @return a double[] of size equal to the number of output attributes
    # * with the values of real attributes. Positions of the vector that doesn't
    # * correspond to a real attribute has no rellevant data.

    def getOutputRealValues(self):
        return self.__realValues[1]

    # end getOutputRealAttributes

    # * Get Output Nominal Values
    # * @return a string[] of size equal to the number of Output attributes
    # * with the values of nominal attributes. Positions of the vector that
    # * doesn't correspond to a nominal attribute has no rellevant data.

    def getOutputNominalValues(self):
        return self.__nominalValues[1]

    # end getOutputNominalValues

    # * Get Output Missing Values
    # * @return a boolean[] of size equal to the number of Output attributes.
    # * A true value in the position i of a vector indicates that the ith
    # * Output value is not known.

    def getOutputMissingValues(self):
        return self.__missingValues[1]

    # end getOutputMissingValues

    #
    # /////////////////////////////////////////////////////////////////////////////
    # //	Functions to get one term of an input or output attribute          //
    # /////////////////////////////////////////////////////////////////////////////
    #
    #
    # /**
    #  * Get Input Real Values
    #  * @return a double with the indicated real input value.
    #  */

    def getInputRealValues(self, pos):
        # print("getInputRealValues, [0][" + str(pos) + "]")
        return self.__realValues[0][pos]

    # end getInputRealAttributes

    '''
     * Get Input Nominal Values
     * @return a string with the indicated nominal input value.
    '''

    def getInputNominalValues(self, pos):
        return self.__nominalValues[0][pos]

    # end getInputNominalValues

    # * It does return the input nominal value at the specified position. The
    # * nominal value is returned as an integer.
    # * @param pos is the position.
    # * @return an int with the nominal value.

    def getInputNominalValuesInt(self, pos):
        return self.__intNominalValues[0][pos]

    # end getInputNominalValues

    '''
     * It does return all the input nominal values.
     * @return an int with the nominal value.
    '''

    def getInputNominalValuesInt(self):
        return self.__intNominalValues[0]

    # end getInputNominalValues

    '''
     * Get Input Missing Values
     * @return a boolean indicating if that input value is missing.
    '''

    def getInputMissingValuesWithPos(self, pos):
        return self.__missingValues[0][pos]

    # end getINputMissingValues

    # * Get Output Real Values
    # * @return a double with the indicated real output value.

    def getOutputRealValues(self, pos):
        return self.__realValues[1][pos]

    # end getOutputRealAttributes

    # * Get Output Nominal Values
    # * @return a string with the indicated nominal output value.

    def getOutputNominalValues(self, pos):
        return self.__nominalValues[1][pos]

    # end getOutputNominalValues

    # * It does return the output value at the specified position
    # * @param pos is the position.
    # * @return an int with the nominal value.

    def getOutputNominalValuesInt(self, pos):
        return self.__intNominalValues[1][pos]

    # end getInputNominalValues

    # * It does return the output value at the specified position
    # * @return an int with the nominal value.

    def getOutputNominalValuesInt(self):
        return self.__intNominalValues[1]

    # end getInputNominalValues

    # * Get Output Missing Values
    # * @return a boolean indicating if that output value is missing.

    def getOutputMissingValues(self, pos):
        return self.__missingValues[1][pos]

    # end getOutputMissingValues

    # /////////////////////////////////////////////////////////////////////////////
    # //	Functions to get all the attributes in a double[]             //
    # /////////////////////////////////////////////////////////////////////////////

    # /**
    #  * It does return all the input values. Doesn't care the type of the attributes.
    #  * Nominal attributes are transformed to an integer, that is codified with a double.
    #  * And integer attributes are codified with a double to. So all the values are
    #  * returnes as doubles.
    #  * @return a double[] with all input values.
    #  */

    def getAllInputValues(self):
        return self.__realValues[0]

    # end getAllInputValues

    # * It does return the normalized values in a double[]. It means that integers are
    # * normalized o [0..N], reals to [0..1] and nominals are transformed to an integer
    # * value between [0..N], where N is the number of values that this nominal can take.
    # * In addition, missing values are represented with a -1 value.

    def getNormalizedInputValues(self):
        norm = [0.0 for x in range(len(self.__realValues[0]))]
        for i in range(0, len(norm)):
            if not self.__missingValues[0][i]:
                norm[i] = Attributes.getInputAttribute(i).normalizeValue(self.__realValues[0][i])
            else:
                norm[i] = -1

        return norm

    # end getNormalizedInputValues

    # * It does return the normalized values in a double[]. It means that integers are
    # * normalized o [0..N], reals to [0..1] and nominals are transformed to an integer
    # * value between [0..N], where N is the number of values that this nominal can take.

    def getNormalizedOutputValues(self):
        norm = [0.0 for x in range(len(self.__realValues[1]))]
        for i in range(0, len(norm)):
            if not self.__missingValues[1][i]:
                norm[i] = Attributes.getOutputAttribute(i).normalizeValue(self.__realValues[1][i])
            else:
                norm[i] = -1.

        return norm

    # end getNormalizedOutputValues

    # /**
    #  * It does return all the output values. Doesn't care the type of the attributes.
    #  * Nominal attributes are transformed to an integer, that is codified with a double.
    #  * And integer attributes are codified with a double to. So all the values are
    #  * returnes as doubles.
    #  * @return a double[] with all output values.
    #  */

    def getAllOutputValues(self):
        return self.__realValues[1]

    # end getAllOutputValues

    # /////////////////////////////////////////////////////////////////////////////
    # //	Functions to set values to an instance                  //
    # /////////////////////////////////////////////////////////////////////////////
    #
    # /**
    #  * It changes the attribute value. If it can't do that, it returns a false
    #  * value.
    #  * @param pos is the attribute that has to be changed
    #  * @param value is the new value
    #  * @return a boolean in false state if the update of the value can't have
    #  * been done.
    #  */

    def setInputNumericValue(self, pos, value):
        at = Attribute(Attributes.getInputAttribute(pos))
        if at.getType() == Attribute.NOMINAL:
            return False
        else:
            if at.isInBounds(value):
                self.__realValues[0][pos] = value
                self.__missingValues[0][pos] = False
                self.__anyMissingValue[0] = False
                for i in range(0, len(self.__missingValues[0])):
                    self.__anyMissingValue[0] |= self.__missingValues[0][i]

            else:
                return False

        return True

    # end setInputNumericValue

    # /**
    #  * It changes the attribute value. If it can't do that, it returns a false
    #  * value.
    #  * @param pos is the attribute that has to be changed
    #  * @param value is the new value
    #  * @return a boolean in false state if the update of the value can't have
    #  * been done.
    #  */

    def setOutputNumericValue(self, pos, value):
        at = Attribute(Attributes.getOutputAttribute(pos))
        if at.getType() == Attribute.NOMINAL:
            return False
        else:
            if at.isInBounds(value):
                self.__realValues[1][pos] = value
                self.__missingValues[1][pos] = False
                self.__anyMissingValue[1] = False
                for i in range(0, len(self.__missingValues[1])):
                    self.__anyMissingValue[1] |= self.__missingValues[1][i]

            else:
                return False

        return True

    # end setInputNumericValue

    # /**
    #  * It set the nominal attribute value to the one passed.
    #  * @param pos is the position of the attribute.
    #  * @param value is the new value.
    #  * @return boolean set to false if the update has not been done.
    #  */

    def setInputNominalValue(self, pos, value):
        print("setInputNominalValue begin......")
        at = Attribute(Attributes.getInputAttribute(pos))
        if at.getType() != Attribute.NOMINAL:
            return False
        else:
            if at.convertNominalValue(value) != -1:
                self.__nominalValues[0][pos] = value
                self.__intNominalValues[0][pos] = at.convertNominalValue(value)
                self.__realValues[0][pos] = self.__intNominalValues[0][pos]
                self.__missingValues[0][pos] = False
                self.__anyMissingValue[0] = False
                for i in range(0, len(self.__missingValues[0])):
                    self.__anyMissingValue[0] |= self.__missingValues[0][i]

            else:
                return False

        return True

    # end setInputNominalValue

    # /**
    #  * It set the nominal attribute value to the one passed.
    #  * @param pos is the position of the attribute.
    #  * @param value is the new value.
    #  * @return boolean set to false if the update has not been done.
    #  */

    def setOutputNominalValue(self, pos, value):
        at = Attribute(Attributes.getOutputAttribute(pos))
        if at.getType() != Attribute.NOMINAL:
            return False
        else:
            if at.convertNominalValue(value) != -1:
                self.__nominalValues[1][pos] = value
                self.__intNominalValues[1][pos] = at.convertNominalValue(value)
                self.__realValues[1][pos] = self.__intNominalValues[0][pos]
                self.__missingValues[1][pos] = False
                self.__anyMissingValue[1] = False
                for i in range(0, len(self.__missingValues[1])):
                    self.__anyMissingValue[1] |= self.__missingValues[1][i]

            else:
                return False

        return True

    # end setOutputNominalValue

    # /////////////////////////////////////////////////////////////////////////////
    # //	General questions about the instance                 //
    # /////////////////////////////////////////////////////////////////////////////

    # /**
    #  * It returns if there is any missing value.
    #  * @return a boolean indicating if there's any missing value.
    #  */

    def existsAnyMissingValue(self):
        return self.__anyMissingValue[0] or self.__anyMissingValue[1]

    # end existsAnyMissingValue

    # /**
    #  * It informs about the existence of missing values in the inputs
    #  * @return a boolean indicating if there's any missing value in the input
    #  */

    def existsInputMissingValues(self):
        return self.__anyMissingValue[0]

    # end existsInputMissingValues

    # /**
    #  * It informs about the existence of missing values in the outputs.
    #  * @return a boolean indicating if there's any missing value in the outputs.
    #  */

    def existsOutputMissingValues(self):
        return self.__anyMissingValue[1]

    # end existsOutputMissingValues

    # /////////////////////////////////////////////////////////////////////////////
    # //	Removing an attribute of the instance                 //
    # /////////////////////////////////////////////////////////////////////////////
    # /**
    #  * It does remove the values of one attribute of the instance.
    #  * @param attToDel is a reference to the attribute to be deleted.
    #  * @param inputAtt is a boolean that indicates if the attribute to be removed
    #  * is an input attribute (otherwise is an output attribute)
    #  * @param whichAtt is the position of the attribute to be deleted.
    #  */

    def removeAttribute(self, attToDel, inputAtt, whichAtt):
        newSize = 0

        # Getting the vector
        index = 0
        if not inputAtt:
            newSize = --self.__numOutputAttributes
            index = 1
        else:
            newSize = --self.__numInputAttributes

        # The number of undefined attributes is increased.
        self.__numUndefinedAttributes += 1

        # It search the absolute position of the attribute to be
        # removed in the list of undefined attributes
        undefPosition = Attributes.searchUndefPosition(attToDel)

        # Reserving auxiliar memory to reconstruct the input or output
        nominalValuesAux = str[newSize]
        intNominalValuesAux = int[newSize]
        realValuesAux = float[newSize]
        missingValuesAux = float[newSize]

        # Reserving auxiliar memory to reconstruct the undefined att's
        nominalValuesUndef = str[self.__numUndefinedAttributes]
        intNominalValuesUndef = int[self.__numUndefinedAttributes]
        realValuesUndef = float[self.__numUndefinedAttributes]
        missingValuesUndef = float[self.__numUndefinedAttributes]

        # Copying the values without the removed attribute
        k = 0
        self.__anyMissingValue[index] = False
        for i in range(0, newSize + 1):
            if i != whichAtt:
                nominalValuesAux[k] = self.__nominalValues[index][i]
                intNominalValuesAux[k] = self.__intNominalValues[index][i]
                realValuesAux[k] = self.__realValues[index][i]
                missingValuesAux[k] = self.__missingValues[index][i]
                if missingValuesAux[k]:
                    self.__anyMissingValue[index] = True
                k += 1

        else:
            nominalValuesUndef[undefPosition] = self.__nominalValues[index][i]
            intNominalValuesUndef[undefPosition] = self.__intNominalValues[index][i]
            realValuesUndef[undefPosition] = self.__realValues[index][i]
            missingValuesUndef[undefPosition] = self.__missingValues[index][i]

        # Copying the rest of the undefined values
        k = 0
        for i in range(0, self.__numUndefinedAttributes):
            if i == undefPosition:
                continue
            nominalValuesUndef[i] = self.__nominalValues[Instance.ATT_NONDEF][k];
            intNominalValuesUndef[i] = self.__intNominalValues[Instance.ATT_NONDEF][k];
            realValuesUndef[i] = self.__realValues[Instance.ATT_NONDEF][k];
            missingValuesUndef[i] = self.__missingValues[Instance.ATT_NONDEF][k];
            k += 1

        # Copying the new vectors without the information of the removed attribute.
        self.__nominalValues[index] = nominalValuesAux
        self.__intNominalValues[index] = intNominalValuesAux
        self.__realValues[index] = realValuesAux
        self.__missingValues[index] = missingValuesAux
        # The undefined attributes
        self.__nominalValues[Instance.ATT_NONDEF] = nominalValuesUndef
        self.__intNominalValues[Instance.ATT_NONDEF] = intNominalValuesUndef
        self.__realValues[Instance.ATT_NONDEF] = realValuesUndef
        self.__missingValues[Instance.ATT_NONDEF] = missingValuesUndef

    # end removeAttribute

    # /////////////////////////////////////////////////////////////////////////////
    # //	Other Instance functions                        //
    # /////////////////////////////////////////////////////////////////////////////

    # /**
    #  * It does return an string with the instance information. The format is the
    #  * same as the read one (keel format). Only are included in the string those
    #  * attributes that are defined as inputs or outputs. So, NON-SPECIFIED-DIRECTION
    #  * attributes are not included to this string.
    #  * The order followed is: first, all input attributes are writen, in the order
    #  * in which they have been read. After that, the output attributes are write.
    #  * This can alter the initial order, but never mind if the output writen
    #  * has the inputs and outputs correctly defined.
    #  * @return a String with the attribute information.
    #  */

    def toStringFunction(self):
        aux = ""
        ending = ","
        for i in range(0, self.__numInputAttributes):
            if i == self.__numInputAttributes - 1 and self.__numOutputAttributes == 0:
                ending = ""
            inputAttrType = Attributes.getInputAttribute(i).getType()
            if inputAttrType == Attribute.NOMINAL:
                aux += self.__nominalValues[0][i]

            if inputAttrType == Attribute.INTEGER:
                aux += str(self.__realValues[0][i])

            if inputAttrType == Attribute.REAL:
                aux += str(self.__realValues[0][i])

            aux += ending

        ending = ","
        for i in range(0, self.__numOutputAttributes):
            if i == self.__numOutputAttributes - 1:
                ending = ""
            outputAttrType = Attributes.getOutputAttribute(i).getType()
            if outputAttrType == Attribute.NOMINAL:
                aux += self.__nominalValues[1][i]

            if outputAttrType == Attribute.INTEGER:
                aux += str(self.__realValues[1][i])

            if outputAttrType == Attribute.REAL:
                aux += str(self.__realValues[1][i])

            aux += ending

        return aux

    # end toString

    #	NEW FUNCTIONS DEFINED FOR NON-STATIC ATTRIBUTES

    def printFunction(self, instAttributes, out):
        out.printInsAttr("    > Inputs: ");
        for i in range(0, self.__numInputAttributes):
            inputAttrType = instAttributes.getInputAttribute(i).getType()
            if inputAttrType == Attribute.NOMINAL:
                out.printInsAttr(self.__nominalValues[Instance.ATT_INPUT][i])

            if inputAttrType == Attribute.INTEGER:
                out.printInsAttr(self.__realValues[Instance.ATT_INPUT][i])

            if inputAttrType == Attribute.REAL:
                out.printInsAttr(self.__realValues[Instance.ATT_INPUT][i])

        out.printInsAttr("\n    > Outputs: ")
        for i in range(0, self.__numOutputAttributes):
            outputAttrType = self.__instAttributes.getOutputAttribute(i).getType()
            if outputAttrType == Attribute.NOMINAL:
                out.printInsAttr(self.__nominalValues[Instance.ATT_OUTPUT][i])

            if outputAttrType == Attribute.INTEGER:
                out.printInsAttr(self.__realValues[Instance.ATT_OUTPUT][i])

            if outputAttrType == Attribute.REAL:
                out.printInsAttr(self.__realValues[Instance.ATT_OUTPUT][i])

        out.printInsAttr("\n    > Undefined: ")
        for i in range(0, self.__numUndefinedAttributes):
            outputAttrType = instAttributes.getOutputAttribute(i).getType()
            if outputAttrType == Attribute.NOMINAL:
                out.printInsAttr(self.__nominalValues[Instance.ATT_OUTPUT][i])
            elif outputAttrType == Attribute.INTEGER:
                out.printInsAttr(self.__realValues[Instance.ATT_OUTPUT][i])
            elif outputAttrType == Attribute.REAL:
                out.printInsAttr(self.__realValues[Instance.ATT_OUTPUT][i])

    # end print

    # * It prints the instance to the specified PrintWriter.
    # * The attribtes order is the same as the one in the
    # * original file.
    # * @param out is the PrintWriter where to print.

    def printAsOriginal(self, instAttributes, out):
        inCount = 0
        outCount = 0
        undefCount = 0
        count = 0
        numAttributes = instAttributes.getNumAttributes()
        for count in range(0, numAttributes):
            at = instAttributes.getAttributeByPos(count)
            directionAttribute = at.getDirectionAttribute()
            if directionAttribute == Attribute.INPUT:
                self.printAttribute(out, Instance.ATT_INPUT, inCount, at.getType())
                inCount += 1

            elif directionAttribute == Attribute.OUTPUT:
                self.printAttribute(out, Instance.ATT_OUTPUT, outCount, at.getType())
                outCount += 1

            elif directionAttribute == Attribute.DIR_NOT_DEF:
                self.printAttribute(out, Instance.ATT_NONDEF, undefCount, at.getType())
                undefCount += 1

        if count + 1 < numAttributes:
            out.printInsAttr(",")

    # end printAsOriginal

    # * It does print the instance information

    def printInstance(self, instAttributes):
        print("  > Inputs (" + self.__numInputAttributes + "): ")

        for i in range(0, self.__numInputAttributes):
            if self.__missingValues[Instance.ATT_INPUT][i]:
                print("?")

            else:
                inputAttributeType = instAttributes.getInputAttribute(i).getType()
                if inputAttributeType == Attribute.NOMINAL:
                    print(self.__nominalValues[Instance.ATT_INPUT][i])

                elif inputAttributeType == Attribute.INTEGER:
                    print(self.__realValues[Instance.ATT_INPUT][i])

                else:
                    if inputAttributeType == Attribute.REAL:
                        print(self.__realValues[Instance.ATT_INPUT][i])

            print("  ")

        print("  > Outputs (" + self.__numOutputAttributes + "): ")
        for i in range(0, self.__numOutputAttributes):
            if self.__missingValues[Instance.ATT_OUTPUT][i]:
                print("?")

            else:
                outputAttrType = self.__instAttributes.getOutputAttribute(i).getType()
                if outputAttrType == Attribute.NOMINAL:
                    print(self.__nominalValues[Instance.ATT_OUTPUT][i])

                elif outputAttrType == Attribute.INTEGER:
                    print(self.__realValues[Instance.ATT_OUTPUT][i])

                elif outputAttrType == Attribute.REAL:
                    print(self.__realValues[Instance.ATT_OUTPUT][i])

        print("  ")

        print("  > Undefined (" + self.__numUndefinedAttributes + "): ")
        for i in range(0, self.__numUndefinedAttributes):
            if self.__missingValues[Instance.ATT_NONDEF][i]:
                print("?")

            else:
                undefinedAttrType = self.__instAttributes.getUndefinedAttribute(i).getType()
                if undefinedAttrType == Attribute.NOMINAL:
                    print(self.__nominalValues[Instance.ATT_NONDEF][i])

                elif undefinedAttrType == Attribute.INTEGER:
                    print(self.__realValues[Instance.ATT_NONDEF][i])

                elif undefinedAttrType == Attribute.REAL:
                    print(self.__realValues[Instance.ATT_NONDEF][i])

            print("  ")

        # end print

    # /**
    #  * Obtains the normalized input attributes from a InstanceAttribute definition
    #  * @param instAttributes The Attributes definition needed to normalize
    #  * @return A new allocated array with the input values normalized
    #  */

    def getNormalizedInputValues(self, instAttributes):
        norm = float[len(self.__realValues[0])]
        for i in range(0, len(norm)):
            if not self.__missingValues[0][i]:
                norm[i] = instAttributes.getInputAttribute(i).normalizeValue(self.__realValues[0][i])
            else:
                norm[i] = -1.

        return norm

    # end getNormalizedInputValues

    # * Obtains the normalized output attributes from a InstanceAttribute definition
    # * @param instAttributes The Attributes definition needed to normalize
    # * @return A new allocated array with the output values normalized

    def getNormalizedOutputValues(self, instAttributes):
        norm = [0.0 for x in range(len(self.__realValues[1]))]
        for i in range(0, len(norm)):
            if not self.__missingValues[1][i]:
                norm[i] = instAttributes.getOutputAttribute(i).normalizeValue(self.__realValues[1][i])
            else:
                norm[i] = -1.

        return norm

    # end getNormalizedOutputValues

    # * Set a new value of a given input attribute in this instance (integer or real)
    # * @param instAttributes The Attributes reference definition
    # * @param pos The position of the input attribute to be changed in instAttributes
    # * @param value The new value

    def setInputNumericValue(self, instAttributes, pos, value):
        at = Attribute(instAttributes.getInputAttribute(pos))
        if at.getType() == Attribute.NOMINAL:
            return False
        else:
            if at.isInBounds(value):
                self.__realValues[0][pos] = value
                self.__missingValues[0][pos] = False
                self.__anyMissingValue[0] = False
                for i in range(0, len(self.__missingValues[0])):
                    self.__anyMissingValue[0] = self.__anyMissingValue[0] + self.__missingValues[0][i]

            else:
                return False

        return True

    # end setInputNumericValue

    # /**
    #  * Set a new value of a given output attribute in this instance (integer or real)
    #  * @param instAttributes The Attributes reference definition
    #  * @param pos The position of the output attribute to be changed in instAttributes
    #  * @param value The new value
    #  * @return true if succeeded, false otherwise

    def setOutputNumericValue(self, instAttributes, pos, value):
        at = Attribute(self.__instAttributes.getOutputAttribute(pos))
        if at.getType() == Attribute.NOMINAL:
            return False
        else:
            if at.isInBounds(value):
                self.__realValues[1][pos] = value
                self.__missingValues[1][pos] = False
                self.__anyMissingValue[1] = False
                for i in range(0, len(self.__missingValues[1])):
                    self.__anyMissingValue[1] = self.__anyMissingValue[1] + self.__missingValues[1][i]

            else:
                return False

        return True

    # end setInputNumericValue

    # * Set a new value of a given input attribute in this instance (nominal)
    # * @param instAttributes The Attributes reference definition
    # * @param pos The position of the input attribute to be changed in instAttributes
    # * @param value The new value

    def setInputNominalValue(self, instAttributes, pos, value):
        at = Attribute(self.__instAttributes.getInputAttribute(pos))
        if at.getType() != Attribute.NOMINAL:
            return False;
        else:
            if at.convertNominalValue(value) != -1:
                self.__nominalValues[0][pos] = value
                self.__intNominalValues[0][pos] = at.convertNominalValue(value)
                self.__realValues[0][pos] = self.__intNominalValues[0][pos]
                self.__missingValues[0][pos] = False
                self.__anyMissingValue[0] = False
                for i in range(0, len(self.__missingValues[0])):
                    self.__anyMissingValue[0] |= self.__missingValues[0][i]

            else:
                return False

        return True

    # end setInputNominalValue

    # * Set a new value of a given output attribute in this instance (nominal)
    # * @param instAttributes The Attributes reference definition
    # * @param pos The position of the output attribute to be changed in instAttributes
    # * @param value The new value
    # * @return true if succeeded, false otherwise

    def setOutputNominalValue(self, instAttributes, pos, value):
        at = Attribute(self.__instAttributes.getOutputAttribute(pos))
        if at.getType() != Attribute.NOMINAL:
            return False
        else:
            if at.convertNominalValue(value) != -1:
                self.__nominalValues[1][pos] = value
                self.__intNominalValues[1][pos] = at.convertNominalValue(value)
                self.__realValues[1][pos] = self.__intNominalValues[0][pos]
                self.__missingValues[1][pos] = False
                self.__anyMissingValue[1] = False
                for i in range(0, len(self.__missingValues[1])):
                    self.__anyMissingValue[1] |= self.__missingValues[1][i]

            else:
                return False

        return True

    # end setOutputNominalValue

    def removeAttribute(self, instAttributes, attToDel, inputAtt, whichAtt):
        newSize = 0

        # Getting the vector
        index = 0
        if not inputAtt:
            newSize = --self.__numOutputAttributes
            index = 1
        else:
            newSize = --self.__numInputAttributes

            # The number of undefined attributes is increased.
            self.__numUndefinedAttributes += 1

        # It search the absolute position of the attribute to be
        # removed in the list of undefined attributes
        undefPosition = instAttributes.searchUndefPosition(attToDel)

        # Reserving auxiliar memory to reconstruct the input or output
        nominalValuesAux = ["" for x in range(newSize)]
        intNominalValuesAux = [0 for x in range(newSize)]
        realValuesAux = [0.0 for x in range(newSize)]
        missingValuesAux = [False for x in range(newSize)]

        # Reserving auxiliar memory to reconstruct the undefined att's
        nominalValuesUndef = ["" for x in range(self.__numUndefinedAttributes)]
        intNominalValuesUndef = [0 for x in range(self.__numUndefinedAttributes)]
        realValuesUndef = [0.0 for x in range(self.__numUndefinedAttributes)]
        missingValuesUndef = [False for x in range(self.__numUndefinedAttributes)]

        # Copying the values without the removed attribute
        k = 0
        self.__anyMissingValue[index] = False
        for i in range(0, newSize + 1):
            if i != whichAtt:
                nominalValuesAux[k] = self.__nominalValues[index][i]
                intNominalValuesAux[k] = self.__intNominalValues[index][i]
                realValuesAux[k] = self.__realValues[index][i]
                missingValuesAux[k] = self.__missingValues[index][i]
                if missingValuesAux[k]:
                    self.__anyMissingValue[index] = True
                k += 1

        else:
            nominalValuesUndef[undefPosition] = self.__nominalValues[index][i]
            intNominalValuesUndef[undefPosition] = self.__intNominalValues[index][i]
            realValuesUndef[undefPosition] = self.__realValues[index][i]
            missingValuesUndef[undefPosition] = self.__missingValues[index][i]

        # Copying the rest of the undefined values
        k = 0
        for i in range(0, self.__numUndefinedAttributes):
            if i == undefPosition:
                continue

            nominalValuesUndef[i] = self.__nominalValues[Instance.ATT_NONDEF][k]
            intNominalValuesUndef[i] = self.__intNominalValues[Instance.ATT_NONDEF][k]
            realValuesUndef[i] = self.__realValues[Instance.ATT_NONDEF][k]
            missingValuesUndef[i] = self.__missingValues[Instance.ATT_NONDEF][k]
            k += 1

        # Copying the new vectors without the information of the removed attribute.
        self.__nominalValues[index] = nominalValuesAux
        self.__intNominalValues[index] = intNominalValuesAux
        self.__realValues[index] = realValuesAux
        self.__missingValues[index] = missingValuesAux
        # The undefined attributes
        self.__nominalValues[Instance.ATT_NONDEF] = nominalValuesUndef
        self.__intNominalValues[Instance.ATT_NONDEF] = intNominalValuesUndef
        self.__realValues[Instance.ATT_NONDEF] = realValuesUndef
        self.__missingValues[Instance.ATT_NONDEF] = missingValuesUndef

    # end removeAttribute

    # * Prints the instance in KEEL format, according to the given Attributes definition
    # * @param instAttributes The reference Attributes definition for printing
    # * @return A new allocated String with the instance in KEEL format (CSV).

    def toString(self, instAttributes):
        aux = ""
        ending = ","
        for i in range(0, self.__numInputAttributes):
            if i == self.__numInputAttributes - 1 and self.__numOutputAttributes == 0:
                ending = ""
            instAttrType = instAttributes.getInputAttribute(i).getType()
            if instAttrType == Attribute.NOMINAL:
                aux += self.__nominalValues[0][i]

            if instAttrType == Attribute.INTEGER:
                aux += str(int(self.__realValues[0][i]))

            if instAttrType == Attribute.REAL:
                aux += str(float(self.__realValues[0][i]))

        aux += ending

        ending = ","
        for i in range(0, self.__numOutputAttributes):
            if i == self.__numOutputAttributes - 1:
                ending = ""
            instOutputAttrType = instAttributes.getOutputAttribute(i).getType()
            if instOutputAttrType == Attribute.NOMINAL:
                aux += self.__nominalValues[1][i]

            elif instOutputAttrType == Attribute.INTEGER:
                aux += str(self.__realValues[1][i])

            elif instOutputAttrType == Attribute.REAL:
                aux += str(float(self.__realValues[1][i]))
            aux += ending

        return aux
    # end toString

    # end of the class Instance
