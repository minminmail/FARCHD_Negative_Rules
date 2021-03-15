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

# /*
#  * Attributes.java
#  *
#  * Created on 20 de junio de 2004, 10:06
#  */
from Help_Classes.Attribute import Attribute


# /**
# * <p>
# * <b> Attributes </b>
# * </p>
# *
# * This class is a static class that, basically, contains a Vector of defined
# * attributes in the train file. Although it keeps all the attributes, it divides
# * them in two groups, the input attributes and the output attributes. It could
# * be that, depending on the @inputs and @outputs defined in the train file, some
# * of the attributes are not valid, so, their values are not loaded to the API
# * dataset. Even in this case, the non-careful attributes information is mantained
# * in this static class.
# *
# * @author Albert Orriols Puig
# * @see Attribute
# * @version keel0.1
# */

class Attributes:
    #
    # /////////////////////////////////////////////////////////////////////////////
    # /////////////// ATTRIBUTES OF THE ATTRIBUTES CLASS //////////////////////////
    # /////////////////////////////////////////////////////////////////////////////
    #
    # /**
    #  * It contains all the attributes definitions.
    #  */
    attributes = []

    # added by rui for set new granularity attributes max and min values.
    granularity_attributes = []

    # /**
    #  * It contains a reference to all input attributes.
    #  */
    inputAttr = []

    # /**
    #  * It contains a reference to all output attributes.
    #  */
    outputAttr = []

    # /**
    #  * It contains a reference to all undefined attributes.
    #  */
    undefinedAttr = []

    # /**
    #  * A flag indicating if the vector contains any nominal attribute.
    #  */
    hasNominal = False
    #
    # /**
    #  * A flag indicating if the vector contains any integer attribute.
    #  */
    hasInteger = False

    # /**
    #  * A flag indicating if the vector contains any real attribute.
    #  */
    hasReal = False

    # /**
    #  * It indicates if there are missing values
    #  */
    hasMissing = False

    # /**
    #  * A vector containing the types of each attribute.
    #  */
    # private static int []type;

    # /**
    #  * String that keeps the relation name
    #  */
    relationName = ""

    # /////////////////////////////////////////////////////////////////////////////
    # ///////////////// METHODS OF THE ATTRIBUTES CLASS ///////////////////////////
    # /////////////////////////////////////////////////////////////////////////////

    # /**
    #  * clearAll
    #  * This method clears all the static members of the class.
    #  * It is used when another data set is wanted to be loaded
    #  */
    def clearAll(self):
        self.attributes = []
        self.inputAttr = []
        self.outputAttr = []
        self.undefinedAttr = []
        self.hasNominal = False
        self.hasInteger = False
        self.hasReal = False
        self.hasMissing = False
        self.relationName = None

    # end clearAll

    # /**
    #  * This method adds an attribute definition.
    #  * @param attr is the new attribute to be added.
    #  */
    def addAttribute(self, attr):
        if not self.isExistent(attr):
            print("In Atrributes class ,addAttribute begin......")
            self.attributes.append(attr)
            attType = attr.getType()
            print("Attribute type is :" + str(attType))
            if (attType == Attribute.NOMINAL):
                self.hasNominal = True
                print("hasNominal is true")
            elif (attType == Attribute.INTEGER):
                self.hasInteger = True
                print("hasInteger is true")
            elif (attType == Attribute.REAL):
                self.hasReal = True
                # print("hasReal is true")

        numberAttribute = len(self.attributes)
        # print("There are " + str(numberAttribute) + "attribute in Attribute class")
        for attr in self.attributes:
            attr_name = attr.getName()
            # print("attr name is :" + str(attr_name))

    # end addAttribute

    #  * The function returns true if the attribute is existent,
    #      will return True else return False otherwise.
    #  */
    def isExistent(self, attr):
        result = False
        for attrExistent in self.attributes:
            if attr.getName() == attrExistent.getName():
                result = True
                break
        return result

    # end isExistent

    # /**
    #  * The function returns if there is any nominal attribute
    #      * @return True if there is any nominal attribute, False otherwise.
    #  */
    def hasNominalAttributes(self):
        return self.hasNominal

    # end hasNominalAttributes

    # /**
    #  * The function returns if there is any integer attribute.
    #      * @return True if there is any integer attribute, False otherwise.
    #  */
    def hasIntegerAttributes(self):
        return self.hasInteger

    # end hasIntegerAttributes

    # /**
    #  * The function returns if there is any real attribute.
    #      * @returnTrue if there is any real attribute, False otherwise.
    #  */
    def hasRealAttributes(self):
        return self.hasReal

    # end hasRealAttributes

    # /**
    #  * The function returns if there is any missing value
    #      * @return if there is any missing value, False otherwise.
    #  */
    def hasMissingValues(self):
        return self.hasMissing

        # end hasMissingValues

    # /**
    #  * It returns the attribute requested.
    #  * @param _name is the name of the attribute.
    #      * @return the attribute requested.
    #  */
    def getAttributeByName(self, _name):
        # print("Begin getAttribute ......")
        size = len(self.attributes)
        stopPos = 0
        for i in range(0, size):
            # print("size of attributes = " + str(size))
            attribute = self.attributes[i]
            if attribute.getName() == _name:
                stopPos = i
                break

        if stopPos == size:
            return None
        return attribute

    # end getAttribute

    # /**
    #  * It does return an array with all attributes
    #      * @return an array with all attributes
    #  */
    def getAttributes(self):
        if len(self.attributes) == 0:
            return None

        attr = [Attribute() for x in range(0, len(self.attributes))]
        for i in range(0, len(attr)):
            attr[i] = self.attributes[i]

        return attr

    # end getAttribute

    # /**
    #  * It returns the input attribute being int the position passed as an argument.
    #  * @param pos is the position of the attribute wanted.
    #      * @return the input attribute being int the position passed as an argument.
    #  */
    def getInputAttribute(self, pos):
        # print("pos is :" + str(pos) + ",self.inputAttr" + str(self.inputAttr))
        if pos < 0 or pos >= len(self.inputAttr):
            # print("Return None for getInputAttribute !!!")
            return None
        else:
            input_att = self.inputAttr[pos]
            return input_att

    # end getInputAttribute

    # /**
    #  * It does return all the input attributes
    #      * @return all the input attributesgetOutputHeader
    #  */
    def getInputAttributes(self):
        if (len(self.inputAttr) == 0):
            return None
        attr = [Attribute() for x in range(0, len(self.inputAttr))]
        for i in range(0, len(attr)):
            attr[i] = self.inputAttr[i]

        return attr

    # end getInputAttribute

    # /**
    #  * It does return an String with the @inputs in keel format.
    #  * @return an string with the @inputs definition  .
    #  */
    def getInputHeader(self):
        aux = "@inputs "
        ending = ","
        for i in range(0, len(self.inputAttr)):
            if i == len(self.inputAttr) - 1:
                ending = ""
            attribute = self.inputAttr[i]
            aux += attribute.getName() + ending
        return aux

    # end getInputHeader

    # /**
    #  * It does return a String with all the input attributes definition in keel
    #  * format. The order of the attributes is the order of lecture.
    #  * @return a String with the input attributes definition.
    #  */
    def getInputAttributesHeader(self):
        aux = ""
        for i in range(0, len(self.inputAttr)):
            # Writting the name and type of the attribute
            aux += self.inputAttr[i].toString() + "\n"

        return aux

    # end getInputAttributesHeader

    #
    # /**
    #  * It does return all the output attributes.
    #      * @return all the output attributes.
    #  */

    def getOutputAttributes(self):
        print("get Output Attributes in Attributes begin.......")
        if len(self.outputAttr) == 0:
            # print("The output attributes are 0:")
            return None
        else:
            attr = [Attribute() for x in range(0, len(self.outputAttr))]
            for i in range(0, len(self.outputAttr)):
                attr[i] = self.outputAttr[i]
            return self.outputAttr

    # end outputAttributes
    # /*
    #  * It returns the output attribute being int the position passed as an argument.
    #  * @param pos is the position of the attribute wanted.
    #      * @return the output attribute being int the position passed as an argument.
    #  */
    def getOutputAttribute(self, pos):
        # print("The pos is " + str(pos))
        # print("len of self.outputAttr is :"+str(len(self.outputAttr)))
        if pos < 0 or pos >= len(self.outputAttr):
            return None
        return self.outputAttr[pos]

    # end getOutputAttribute

    # /**
    #  * It does return an String with the @outputs in keel format.
    #  * @return an string with the @outputs definition  .
    #  */
    def getOutputHeader(self):
        aux = "@outputs "
        ending = ","
        for i in range(0, len(self.outputAttr)):
            if i == len(self.outputAttr) - 1:
                ending = " "
            aux = aux + self.outputAttr[i].getName() + ending

        return aux

    # end getOutputHeader

    # /**
    #  * It does return a String with all the output attributes definition in keel
    #  * format. The order of the attributes is the order of lecture.
    #  * @return a String with the output attributes definition.
    #  */
    def getOutputAttributesHeader(self):
        aux = ""
        for i in range(0, len(self.outputAttr)):
            # Writting the name and type of the attributeoutputAttr
            aux = aux + self.outputAttr[i].toString() + "\n"
        # print("getOutputAttributesHeader, aux =" + aux)
        return aux

    # end getOutputAttributesHeader

    # /**
    #  * It returns the undefined attribute being int the position passed as an argument.
    #  * @param pos is the position of the attribute wanted.
    #      * @return the undefined attribute being int the position passed as an argument.
    #  *
    #  */
    def getUndefinedAttribute(self, pos):
        if pos < 0 or pos >= len(self.undefinedAttr):
            return None
        return self.undefinedAttr[pos]

    # end getUndefinedAttribute

    # /**
    #  * It does return all the undefined attributes
    #      * @return all the undefined attributes
    #  */
    def getUndefinedAttributes(self):
        if len(self.undefinedAttr) == 0:
            return None
        attr = [Attribute() for x in range(0, len(self.undefinedAttr))]
        for i in range(0, attr.length):
            attr[i] = self.undefinedAttr[i]

        return attr

    # end getUndefinedAttributes

    # /**
    #  * It does return a String with all the undefined attributes definition
    #  * in keel format. The order of the attributfile_lines is emptyes is the order of lecture.
    #  * @return a String with the input attributes definition.
    #  */
    def getUndefinedAttributesHeader(self):
        aux = ""
        for i in range(0, len(self.undefinedAttr)):
            # Writting the name and type of the attribute
            aux += self.undefinedAttr[i].toString() + "\n"
        # print("getUndefinedAttributesHeader, aux = " + aux)
        return aux

    # end getUndefinedAttributesHeader

    # /**
    #  * It returns the attribute being int the position passed as an argument.
    #  * @param pos is the position of the attribute wanted.
    #      * @return the attribute being int the position passed as an argument.
    #  *

    def getAttributeByPos(self, pos):
        lengthAtt = len(self.attributes)
        # print("The size of attribute array is :" + str(lengthAtt))
        # print("The pos given is :" + str(pos))
        if pos < lengthAtt:
            attStr = self.attributes[pos]
            # print("Return :" + str(attStr))
            return attStr
        else:
            print(" Return None !!! pos is bigger than array length, will cause out of index error .")
            return None

    # end getAttribute

    # /**
    #  * It return the total number of attributes in the API
    #  * @return an int with the number of attributes
    #  */
    def getNumAttributes(self):
        return len(self.attributes)

    # end getNumAttributes

    # /**
    #  * It return the  number of input attributes in the API
    #  * @return an int with the number of attributes
    #  */
    def getInputNumAttributes(self):
        return len(self.inputAttr)

    # end getInputNumAttributes

    # /**
    #  * It return the number of output attributes in the API
    #  * @return an int with the number of attributes
    #  */
    def getOutputNumAttributes(self):
        # print("begin getOutputNumAttributes ......")

        number_of_outputAttr = len(self.outputAttr)
        # print("number_of_outputAttr : " + str(number_of_outputAttr))
        return number_of_outputAttr

    # end getOutputNumAttributes

    # /**
    #  * It return the number of undefined attributes in the API
    #  * @return an int with the number of attributes
    #  */
    def getUndefinedNumAttributes(self):
        return len(self.undefinedAttr)

    # end getUndefinedNumAttributes

    # /**
    #  * It returns all the attribute names in the dataset except these ones
    #  * that are already in the vector v.
    #  * @param v is a vector with the exceptions
    #  * @return a Vector with the rest of attribute names.
    #  */
    def getAttributesExcept(self, vector):
        restAt = []
        for i in range(0, len(self.attributes)):
            attName = self.attributes[i].getName()
            if attName not in vector:
                restAt.append(attName)

        return restAt

    # end getAttributesExcept

    # /**
    #  * It organizes the whole number of attributes to input, output, and
    #  * "no-direction" attributes.
    #  * @param inAttNames  is a vector with the names of all input  attributes.
    #  * @param outAttNames is a vector with the names of all output attributes.
    #  */
    def setOutputInputAttributes(self, inAttNames, outAttNames):

        attName = ""
        att = None
        # for inAtt in inAttNames:
        # print("inAtt name inside inAttNames is:" + inAtt)
        # for outAtt in outAttNames:
        # print("outAtt name inside outAttNames is:" + outAtt)

        for i in range(0, len(self.attributes)):
            att = self.attributes[i]

            attName = att.getName()
            attName = attName.strip()
            # print("attName is:" + str(attName))

            if attName in inAttNames:
                # print("attName in inAttNames")
                if not self.hasSameAttributeName(attName, self.inputAttr):
                    # print("add in input attribute list, attName is:" + attName)
                    att.setDirectionAttribute(Attribute.INPUT)
                    self.inputAttr.append(self.attributes[i])
            elif attName in outAttNames:
                if not self.hasSameAttributeName(attName, self.outputAttr):
                    # print("add in out attribute list, attName is:" + attName)
                    att.setDirectionAttribute(Attribute.OUTPUT)
                    self.outputAttr.append(self.attributes[i])
            elif not self.hasSameAttributeName(attName, self.undefinedAttr):
                # print("add in undefinedAttr attribute list, attName is:" + attName)
                self.undefinedAttr.append(self.attributes[i])

        # Finally, making some statistics
        self.hasNominal = False
        self.hasInteger = False
        self.hasReal = False

        for index in range(0, 2):                                         
            if index == 0:
                iterations = len(self.inputAttr)
            else:
                iterations = len(self.outputAttr)

            for i in range(0, iterations):                                                                                                                                                                                                                                                                                                               
                if index == 0:
                    att = self.inputAttr[i]
                else:
                    att = self.outputAttr[i]

                type = att.getType()
                if type == Attribute.NOMINAL:
                    self.hasNominal = True
                elif type == Attribute.INTEGER:
                    self.hasInteger = True
                elif type == Attribute.REAL:
                    self.hasReal = True

    # end setOutputInputAttributes

    # /**
    #  * This method checks if all the input names vector corresponds with
    #  * all the attributes in input vector. If not, it returns a false. It
    #  * is used in a test to check that the definition of input attributes
    #  * is the same as the definition made in train.
    #  * @param outputNames is a vector with all input attribute names.
    #  */
    def hasSameAttributeName(self,attrName, attr_list):
        hasSame = False
        # print("attrName ==" + attrName)
        for item_in_list in attr_list:
            name = item_in_list.getName()
            # print("item_in_list.getName() ==" + name)

            if item_in_list.getName() == attrName:
                hasSame = True
                break

        # print(" return hasSame ==" + str(hasSame))
        return hasSame

    def areAllDefinedAsInputs(self, inputNames):
        if len(inputNames) != len(self.inputAttr):
            return False
        for i in range(0, len(self.inputAttr)):
            name = self.inputAttr[i].getName()
            if name not in inputNames:
                return False

        return True

    # end areAllDefinedAsInputs

    # /**
    #  * This method checks if all the output names vector corresponds with
    #  * all the attributes in output vector. If not, it returns a false. It
    #  * is used in a test to check that the definition of output attributes
    #  * is the same as the definition made in train.
    #  * @param outputNames is a vector with all output attribute names.
    #      * @return True if all the output names vector corresponds with
    #  * all the attributes in output vector.
    #  */

    def areAllDefinedAsOutputs(self, outputNames):
        if len(outputNames) != len(self.outputAttr):
            return False

        for i in range(0, len(self.outputAttr)):
            name = self.outputAttr[i].getName()
            if name not in outputNames:
                return False

        return True

    # end areAllDefinedAsOutputs

    # /**
    #  * It sets the relation name.
    #  * @param rel is the name to be set to the relationName
    #  */
    def setRelationName(self, rel):
        self.relationName = rel

    # end setRelationName

    # /**
    #  * It gets the relation name.
    #  * @return an String with the realtion name.
    #  */
    def getRelationName(self):
        return self.relationName

    # end relationName

    # /**
    #  * It does remove an attribute. Removing an attribute only implies, in terms
    #  * of Attribute static class, to take it out from the input/output attributes
    #  * list, but it will never be removed from the attributes general list. So
    #  * it will be placed as a NON-SPECIFIED attribute, as it wasn't declared in
    #  * neither @inputs and @outputs definition.
    #  * @param inputAtt is a boolean that indicates if the attribute to be removed
    #  * is an input attribute
    #  * @param whichAtt is an integer that indicates the position of the attribute
    #  * to be removed.
    #  * @return a boolean that will be false if the attribute hasn't been found.
    #  */
    def removeAttribute(self, inputAtt, whichAtt):
        atToDel = None
        if inputAtt and (whichAtt >= len(self.inputAttr) or whichAtt < 0):
            return False
        if not inputAtt and (whichAtt >= len(self.outputAttr) or whichAtt < 0):
            return False
        if inputAtt:
            # inputAttribute
            atToDel = self.inputAttr[whichAtt]
            atToDel.setDirectionAttribute(Attribute.DIR_NOT_DEF)
            del self.inputAttr[whichAtt]

        else:  # output attribute
            atToDel = self.outputAttr[whichAtt]
            atToDel.setDirectionAttribute(Attribute.DIR_NOT_DEF)
            del self.outputAttr[whichAtt]

        # We get the position where it has to go in the undefined attributes vector.
        undef_position = self.searchUndefPosition(atToDel)
        self.undefinedAttr.insert(undef_position, atToDel)

        self.hasNominal = False
        self.hasInteger = False
        self.hasReal = False
        for index in range(0, 2):
            if index == 0:
                iterations = len(self.inputAttr)
            else:
                iterations = len(self.outputAttr)

            for i in range(0, iterations):

                if index == 0:
                    att = self.inputAttr[i]
                else:
                    att = self.outputAttr[i]

                type = att.getType()
                if type == Attribute.NOMINAL:
                    self.hasNominal = True
                elif type == Attribute.INTEGER:
                    self.hasInteger = True

                elif type == Attribute.REAL:
                    self.hasReal = True

        return True

    # end removeAttribute

    #
    # /**
    #  * It does search the relative position of the input/output attribute
    #  * 'whichAtt' in the list of indefined attributes.
    #  * @param attToDel is an Attribute reference to the attribute that has to
    #  * be deleted.
    #  * @return an int with the relative position.
    #  */
    def searchUndefPosition(self, attToDel):
        undefCount = 0
        count = 0

        att_aux = self.attributes[count]
        while attToDel != att_aux:
            if att_aux.getDirectionAttribute() == Attribute.DIR_NOT_DEF:
                undefCount += 1

            count += 1
            att_aux = self.attributes[count]

        return undefCount

    # end searchUndefPosition
    #
    # /**
    #  * It does initializes the statistics to make the statistics. It only
    #  * works for classifier Datasets (only one output).
    #  */

    def initStatistics(self):
        # print("In Attributes initStatistics begin.....")
        outputAttNumber = len(self.outputAttr)
        # print("In initStatistics of Attributes, the outputAttNumber is " + str(outputAttNumber))
        if outputAttNumber != 1:
            return
        for out_put_att in self.outputAttr:
            name = out_put_att.getName()
            # print("out_put_att is :" + name)
        # print("outputAttr[0]" + str(self.outputAttr[0]))
        classNumber = self.outputAttr[0].getNumNominalValues()
        # print("inside initStatistics the classNumber is:" + str(classNumber))
        # If the outpout_put_att isut attribute has not been defined as a nominal or it has not
        # any value in the nominal list, the initalization is aborted.
        if classNumber <= 0:
            # print("class Number is smaller than 0, return")
            return
        else:
            input_attr_length = len(self.inputAttr)
            # print("class Number is bigger than 0, input_attr_length = " + str(input_attr_length))
            for i in range(0, input_attr_length):
                # print("Call Attribute.initStatisticsTwo in Attributes initStatistics......classNumber = " + str(classNumber))
                (self.inputAttr[i]).initStatisticsTwo(classNumber)

        # end initStatistics

    # /**
    #  * It does finish the statistics
    #  */
    def finishStatistics(self):
        if len(self.outputAttr) != 1:
            return

        for i in range(0, len(self.inputAttr)):
            (self.inputAttr[i]).finishStatistics()

    # end finishStatistics

    # /**
    #  * It does # print the attributes information
    #  */
    def printAttributes(self):
        # print("@relation = " + self.relationName)
        for i in range(0, len(self.attributes)):
            att = self.attributes[i]
            if att.getDirectionAttribute() == Attribute.INPUT:
                print(" > INPUT ATTRIBUTE:")
            elif att.getDirectionAttribute() == Attribute.OUTPUT:
                print("> OUTPUT ATTRIBUTE:")
            else:
                print("> UNDEFINED ATTRIBUTE:")

            att.printAttr()

    # end print

    # end of Attributes class
