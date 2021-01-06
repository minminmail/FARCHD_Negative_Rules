'''
/***********************************************************************

    This file is part of KEEL-software, the Data Mining tool for regression,
    classification, clustering, pattern mining and so on.

    Copyright (C) 2004-2010

    F. Herrera (herrera@decsai.ugr.es)
    L. S谩nchez (luciano@uniovi.es)
    J. Alcal谩-Fdez (jalcala@decsai.ugr.es)
    S. Garc铆a (sglopez@ujaen.es)
    A. Fern谩ndez (alberto.fernandez@ujaen.es)
    J. Luengo (julianlm@decsai.ugr.es)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see http://www.gnu.org/licenses/

**********************************************************************/

/*
 * InstanceAttributes.java
 *
 * Created on 20 de junio de 2004, 10:06
 */
 '''

# * <p>
# * <b> InstanceAttributes </b>
# * </p>
# *
# * This class contains the information of all the attributes in the dataset.
# * It stores the same information in Attributes, but it is not defined as static.
# *
# * @author Albert Orriols Puig
# * @see Attribute
# * @version keel0.1
# */


from Help_Classes import Attribute, Attributes


class InstanceAttributes:
    # /////////////////////////////////////////////////////////////////////////////
    # /////////////// ATTRIBUTES OF THE ATTRIBUTES CLASS //////////////////////////
    # /////////////////////////////////////////////////////////////////////////////
    #
    # /**
    #  * It contains all the attributes definitions.
    #  */

    __attributes = []

    # It contains a reference to all input attributes.

    __inputAttr = []

    # It contains a reference to all output attributes.

    __outputAttr = []

    # It contains a reference to all undefined attributes.

    __undefinedAttr = []

    # A flag indicating if the vector contains any nominal attribute.

    __hasNominal = None

    # A flag indicating if the vector contains any integer attribute.

    __hasInteger = None

    # A flag indicating if the vector contains any real attribute.

    __hasReal = None

    # A vector containing the types of each attribute.

    # private static int []type;

    # String that keeps the relation name

    __relationName = ""

    # /////////////////////////////////////////////////////////////////////////////
    # ///////////////// METHODS OF THE ATTRIBUTES CLASS ///////////////////////////
    # /////////////////////////////////////////////////////////////////////////////

    # /**
    #  * InstanceAttributes
    #  *
    #  * Class constructor. It reserve memory to allocate the attributes
    #  */

    def __init__(self):
        self.__attributes = []
        self.__inputAttr = []
        self.__outputAttr = []
        self.__undefinedAttr = []
        self.__hasNominal = False
        self.__hasInteger = False
        self.__hasReal = False
        self.__relationName = ""

    # end clearAll

    def __init_two(self, instance_attr):
        self.__attributes = instance_attr.__attributes
        self.__inputAttr = instance_attr.__inputAttr
        self.__outputAttr = instance_attr.__outputAttr
        self.__undefinedAttr = instance_attr.__undefinedAttr
        self.__hasInteger = instance_attr.__hasInteger
        self.__hasNominal = instance_attr.__hasNominal
        self.__hasReal = instance_attr.__hasReal
        self.__relationName = str(instance_attr.__relationName)

    # /**
    #  * copyStaticAttributes
    #  *
    #  * It copies the attributes definition statically stored in Attributes class
    #  /

    def copyStaticAttributes(self):

        self.__attributes = []
        self.__inputAttr = []
        self.__outputAttr = []
        self.__undefinedAttr = []

        for i in range(0, len(Attributes.attributes)):
            self.__attributes.add(Attributes.attributes[i])
        for i in range(0, len(Attributes.inputAttr)):
            self.__inputAttr.add(Attributes.inputAttr[i])

        for i in range(0, len(Attributes.outputAttr)):
            self.__outputAttr.add(Attributes.outputAttr[i])
        for i in range(0, len(Attributes.undefinedAttr)):
            self.__undefinedAttr.add(Attributes.undefinedAttr[i])

        self.__hasNominal = Attributes.hasNominal
        self.__hasInteger = Attributes.hasInteger
        self.__hasReal = Attributes.hasReal
        self.__relationName = Attributes.relationName

    # end copyStaticAttributes

    # /**
    #  * This method adds an attribute definition.
    #  * @param attr is the new attribute to be added.
    #  */

    def addAttribute(self, attr):
        self.__attributes.append(attr)
        if attr.getDirectionAttribute() == Attribute.INPUT:
            self.__inputAttr.append(attr)
        if attr.getDirectionAttribute() == Attribute.OUTPUT:
            self.__outputAttr.append(attr)
        if attr.getDirectionAttribute() == Attribute.DIR_NOT_DEF:
            self.__undefinedAttr.append(attr)
        if attr.getType() == Attribute.NOMINAL:
            self.__hasNominal = True
        if attr.getType() == Attribute.INTEGER:
            self.__hasInteger = True
        if attr.getType() == Attribute.REAL:
            self.__hasReal = True

    # end addAttribute

    # The function returns if there is any nominal attribute

    def hasNominalAttributes(self):
        return self.__hasNominal

    # end hasNominalAttributes

    # /**
    #  * The function returns if there is any integer attribute.
    #  */

    def hasIntegerAttributes(self):
        return self.__hasInteger

    # end hasIntegerAttributes

    # The function returns if there is any real attribute.

    def hasRealAttributes(self):
        return self.__hasReal

    # end hasRealAttributes

    # /**
    #  * It returns the attribute requested.
    #  * @param _name is the name of the attribute.
    #  */

    def getAttribute(self, _name):
        i = 0
        for i in range(0, len(self.__attributes)):
            if Attribute(self.__attributes[i]).getName() == _name:
                break

        if i == len(self.__attributes):
            return None
        return Attribute(self.__attributes[i])

    # end getAttribute

    # It does return an array with all attributes

    def getAttributes(self):
        if len(self.__attributes) == 0:
            return None
        attr = Attribute[len(self.__attributes)]
        for i in range(0, len(attr)):
            attr[i] = Attribute(self.__attributes[i])

    # end getAttribute

    # * It returns the input attribute being int the position passed as an argument.
    # * @param pos is the position of the attribute wanted.

    def getInputAttribute(self, pos):
        if pos < 0 or pos >= len(self.__inputAttr):
            return None
        return Attribute(self.__inputAttr[pos])

    # end getInputAttribute

    # It does return all the input attributes

    def getInputAttributes(self):
        if len(self.__inputAttr) == 0:
            return None
        attr = Attribute[len(self.__inputAttr)]
        for i in range(0, len(attr)):
            attr[i] = Attribute(self.__inputAttr[i])
            return attr

    # end getInputAttribute

    # /**
    #  * It does return an String with the @inputs in keel format.
    #  * @return an string with the @inputs definition  .
    #  */
    def getInputHeader(self):
        aux = "@inputs "
        ending = ","
        inputLength = len(self.__inputAttr)
        for i in range(0, inputLength):
            if i == (inputLength - 1):
                ending = ""
            aux += (Attribute(self.__inputAttr[i])).getName() + ending

        return aux

    # end getInputHeader

    # /**
    #  * It does return a String with all the input attributes definition in keel
    #  * format. The order of the attributes is the order of lecture.
    #  * @return a String with the input attributes definition.
    #  */

    def getInputAttributesHeader(self):
        aux = "";
        for i in range(0, len(self.__inputAttr)):
            # Writting the name and type of the attribute
            aux += self.__inputAttr[i].toString() + "\n";

        return aux

    # end getInputAttributesHeader

    # /**
    #  * It does return all the output attributes.
    #  */

    def getOutputAttributes(self):
        if len(self.__outputAttr) == 0:
            return None
        attr = Attribute[len(self.__outputAttr)]
        for i in range(0, len(attr)):
            attr[i] = self.__outputAttr[i]

        return attr

    # end outputAttributes

    # /**
    #  * It returns the output attribute being int the position passed as an argument.
    #  * @param pos is the position of the attribute wanted.
    #  */

    def getOutputAttribute(self, pos):
        if pos < 0 or pos >= len(self.__outputAttr):
            return None
        return Attribute(self.__outputAttr[pos])

    # end getOutputAttribute

    # /**
    #  * It does return an String with the @outputs in keel format.
    #  * @return an string with the @outputs definition  .
    #  */

    def getOutputHeader(self):
        aux = "@outputs ";
        ending = ","
        out_put_att_length = len(self.__outputAttr)
        for i in range(0, out_put_att_length):
            if i == out_put_att_length - 1:
                ending = ""
            aux += (Attribute(self.outputAttr[i]).getName()) + ending;

        return aux

    # end getOutputHeader

    # /**
    #  * It does return a String with all the output attributes definition in keel
    #  * format. The order of the attributes is the order of lecture.
    #  * @return a String with the output attributes definition.
    #  */

    def getOutputAttributesHeader(self):
        aux = ""
        for i in range(0, len(self.__outputAttr)):
            # Writting the name and type of the attribute
            aux += self.__outputAttr[i].toString() + "\n"

        return aux

    # end getOutputAttributesHeader

    # /**
    #  * It returns the undefined attribute being int the position passed as an argument.
    #  * @param pos is the position of the attribute wanted.
    #  */

    def getUndefinedAttribute(self, pos):
        if pos < 0 or pos >= len(self.__undefinedAttr):
            return None
        return Attribute(self.__undefinedAttr[pos])

    # end getUndefinedAttribute

    # /**
    #  * It does return all the undefined attributes
    #  */

    def getUndefinedAttributes(self):
        if len(self.__undefinedAttr) == 0:
            return None
        attr = Attribute[len(self.__undefinedAttr)]
        for i in range(0, attr.length):
            attr[i] = Attribute(self.__undefinedAttr[i])

        return attr
        # end getUndefinedAttributes

    # /**
    #  * It does return a String with all the undefined attributes definition
    #  * in keel format. The order of the attributes is the order of lecture.
    #  * @return a String with the input attributes definition.
    #  */

    def getUndefinedAttributesHeader(self):
        aux = ""
        for i in range(0, len(self.__undefinedAttr)):
            # Writting the name and type of the attribute
            aux += self.__undefinedAttr[i].toString() + "\n";

        return aux

    # end getUndefinedAttributesHeader

    # /**
    #  * It returns the attribute being int the position passed as an argument.
    #  * @param pos is the position of the attribute wanted.
    #  */

    def getAttribute(self, pos):
        return Attribute(self.__attributes[pos]);

    # end getAttribute

    # /**
    #  * It return the total number of attributes in the API
    #  * @return an int with the number of attributes
    #  */

    def getNumAttributes(self):
        return len(self.__attributes)

    # end getNumAttributes

    # /**
    #  * It return the  number of input attributes in the API
    #  * @return an int with the number of attributes
    #  */
    def getInputNumAttributes(self):
        return len(self.__inputAttr)

    # end getInputNumAttributes

    # /**
    #  * It return the number of output attributes in the API
    #  * @return an int with the number of attributes
    #  */

    def getOutputNumAttributes(self):
        return len(self.__outputAttr)

    # end getOutputNumAttributes

    # /**
    #  * It return the number of undefined attributes in the API
    #  * @return an int with the number of attributes
    #  */

    def getUndefinedNumAttributes(self):
        return len(self.__undefinedAttr)

    # end getUndefinedNumAttributes

    # /**
    #  * It returns all the attribute names in the dataset except these ones
    #  * that are already in the vector v.
    #  * @param v is a vector with the exceptions
    #  * @return a Vector with the rest of attribute names.
    #  */

    def getAttributesExcept(self, v):
        restAt = []
        for i in range(0, len(self.__attributes)):
            attName = Attribute(self.__attributes[i]).getName()
            if attName not in v:
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
        i = 0
        attName = ""
        att = Attribute()

        for i in range(0, len(self.__attributes)):
            att = Attribute(self.__attributes[i])
            attName = att.getName()
            if attName in inAttNames:
                att.setDirectionAttribute(Attribute.INPUT)
                self.__inputAttr.append(self.__attributes[i])
            elif attName in outAttNames:
                att.setDirectionAttribute(Attribute.OUTPUT)
                self.__outputAttr.append(self.__attributes[i])
            else:
                self.__undefinedAttr.append(self.__attributes[i])

        # Finally, making some statistics
        self.__hasNominal = False
        self.__hasInteger = False
        self.__hasReal = False

        for index in range(0, 2):
            if index == 0:
                iterations = len(self.__inputAttr)
            else:
                iterations = len(self.__outputAttr)

            for i in range(0, iterations):

                if index == 0:
                    att = Attribute(self.__inputAttr[i])
                else:
                    att = Attribute(self.__outputAttr[i])
                if att.getType() == Attribute.NOMINAL:

                    self.__hasNominal = True

                elif att.getType() == Attribute.INTEGER:
                    self.__hasInteger = True

                elif att.getType() == Attribute.REAL:
                    self.__hasReal = True

    # end setOutputInputAttributes

    # /**
    #  * This method checks if all the input names vector corresponds with
    #  * all the attributes in input vector. If not, it returns a false. It
    #  * is used in a test to check that the definition of input attributes
    #  * is the same as the definition made in train.
    #  * @param outputNames is a vector with all input attribute names.
    #  */

    def areAllDefinedAsInputs(self, inputNames):
        if len(inputNames) != len(self.__inputAttr):
            return False

        for i in range(0, len(self.__inputAttr)):
            input_name = Attribute(self.__inputAttr[i]).getName()
            if input_name not in inputNames:
                return False
        return True

    # end areAllDefinedAsInputs

    # /**
    #  * This method checks if all the output names vector corresponds with
    #  * all the attributes in output vector. If not, it returns a false. It
    #  * is used in a test to check that the definition of output attributes
    #  * is the same as the definition made in train.
    #  * @param outputNames is a vector with all output attribute names.
    #  */

    def areAllDefinedAsOutputs(self, outputNames):
        if outputNames.size() != len(self.__outputAttr):
            return False

        for i in range(0, len(self.__outputAttr)):
            out_put_name = Attribute(self.__outputAttr[i]).getName()
            if out_put_name not in outputNames:
                return False

        return True

    # end areAllDefinedAsOutputs

    # /**
    #  * It sets the relation name.
    #  * @param rel is the name to be set to the relationName
    #  */

    def setRelationName(self, rel):
        self.__relationName = rel

    # end setRelationName

    # /**
    #  * It gets the relation name.
    #  * @return an String with the realtion name.
    #  */
    #

    def getRelationName(self):
        return self.__relationName

    # end relationName
    #
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
        atToDel = None;
        if inputAtt and (whichAtt >= len(self.__inputAttr) or whichAtt < 0):
            return False;
        if self.__inputAtt and (whichAtt >= len(self.__outputAttr) or whichAtt < 0):
            return False;

        if inputAtt:
            # inputAttribute
            atToDel = Attribute(self.__inputAttr[whichAtt])
            atToDel.setDirectionAttribute(Attribute.DIR_NOT_DEF)
            self.__inputAttr.removeElementAt(whichAtt);

        else:  # output attribute
            atToDel = Attribute(self.__outputAttr[whichAtt])
            atToDel.setDirectionAttribute(Attribute.DIR_NOT_DEF)
            self.__outputAttr.removeElementAt(whichAtt)

        # We get the position where it has to go in the undefined attributes vector.
        self.__undefPosition = self.searchUndefPosition(atToDel)
        self.__undefinedAttr.insertElementAt(atToDel, self.__undefPosition)

        self.__hasNominal = False
        self.__hasInteger = False
        self.__hasReal = False
        for index in (0, 2):
            iterations = 0
            if index == 0:
                iterations = len(self.__inputAttr)
            else:
                iterations = len(self.__outputAttr)
            for i in range(0, iterations):
                if index == 0:
                    att = Attribute(self.__inputAttr[i])
                else:
                    att = Attribute(self.__outputAttr[i])
            attTypeHere = att.getType()
            if attTypeHere == Attribute.NOMINAL:

                self.__hasNominal = True

            elif attTypeHere == Attribute.INTEGER:
                self.__hasInteger = True

            elif attTypeHere == Attribute.REAL:
                self.__hasReal = True

        return True

    # end removeAttribute

    # * It does search the relative position of the input/output attribute
    # * 'whichAtt' in the list of indefined attributes.
    # * @param attToDel is an Attribute reference to the attribute that has to
    # * be deleted.
    # * @return an int with the relative position.
    # */

    def searchUndefPosition(self, attToDel):
        undefCount = 0
        count = 0

        att_aux = Attribute(self.__attributes[count])
        while attToDel != att_aux:
            if att_aux.getDirectionAttribute() == Attribute.DIR_NOT_DEF:
                undefCount = undefCount + 1

            count = count + 1
            att_aux = Attribute(self.__attributes[count])

        return undefCount

    # end searchUndefPosition

    # /**
    #  * It does initializes the statistics to make the statistics. It only
    #  * works for classifier Datasets (only one output).
    #  */

    def initStatistics(self):
        if len(self.__outputAttr) != 1:
            return

        classNumber = self.__outputAttr[0].getNumNominalValues(Attribute)
        # If the output attribute has not been defined as a nominal or it has not
        # any value in the nominal list, the initalization is aborted.
        if classNumber <= 0:
            return

        for i in range(0, len(self.__inputAttr)):
            self.__inputAttr[i].initStatisticsTwo(classNumber)

    # end initStatistics

    # /**
    #  * It does finish the statistics
    #  */

    def finishStatistics(self):
        if len(self.__outputAttr) != 1:
            return

        for i in range(0, len(self.__inputAttr)):
            Attribute(self.__inputAttr[i]).finishStatistics()

    # //end finishStatistics

    # /**
    #  * It does print the attributes information
    #  */

    def printInsAttr(self):
        # print("@relation = " + self.__relationName)
        # print("Number of attributes: " + str(len(self.__attributes)))

        for i in range(0, len(self.__attributes)):
            att = Attribute(self.__attributes[i])
            if att.getDirectionAttribute() == Attribute.INPUT:
                print("  > INPUT ATTRIBUTE:     ")
            elif att.getDirectionAttribute() == Attribute.OUTPUT:
                print("  > OUTPUT ATTRIBUTE:    ")
            else:
                print("  > UNDEFINED ATTRIBUTE: ")

            att.printAttr()

        # end print
        # end of Attributes class
