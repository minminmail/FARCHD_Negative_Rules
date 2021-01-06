#
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

# /*
#  * Parser.java
#  *
#  * Created on 24 de enero de 2005, 10:43
#  */

# /**
#  * <p>
#  * <b> InstanceParser </b>
#  * </p>
#  * This class is a parser for the instances. It reads an instance (a line
#  * from the data file) and returns it. It also mantain some information as
#  * the relation name.
#  *
#  * @author  Albert Orriols Puig
#  * @version keel0.1
#  *
#  */

class InstanceParser:
    # /////////////////////////////////////////////////////////////////////////////
    # ////////////////// ATTRIBUTES OF THE PARSER CLASS ///////////////////////////
    # /////////////////////////////////////////////////////////////////////////////
    #
    # /**
    #  * A Buffered Reader to the DB input file.
    #  */

    #
    # /**
    #  * A flag indicating if the DB is a train or a test DB. The difference between
    #  * them is that a test DB doesn't modify any parameter definition.
    #  */

    __isTrain = None
    # It counts the attribute number.
    __attributeCount = 0
    # String where de file header is stored
    __header = ""
    # String where the relation name is stored
    __relation = ""
    # Counter of the line
    lineCounter = 0
    file = None
    # added for granularity rule

    data_row_array = []
    data_row_counter = 0


    # /////////////////////////////////////////////////////////////////////////////
    # /////////////////// METHODS OF THE PARSER CLASS /////////////////////////////
    # /////////////////////////////////////////////////////////////////////////////
    # /**
    #  * It does create a new instance of ParserARFF.
    #  * @param fileName is the file name of the DB file.
    #  * @param _isTrain is a flag that indicates if the DB is for a train.
    #  */
    #

    def __init__(self, fileName, _isTrain):
        try:
            print("In init method of InstanceParser begin......")
            self.file = open(fileName, "r")
            print("In init of InstanceParser, set file =" + str(fileName))
            # print(self.file.read())
            self.lineCounter = 0
        except Exception as error:
            print("The exception in init of InstanceParse is: " + format(error))
            exit(1)

        self.__isTrain = _isTrain
        self.__attributeCount = 0

    # end of Parser constructor

    def init_for_granularity_parser(self, data_array, _isTrain):
        try:
            print("In init of init_for_granularity_parser, set file =")
            self.data_row_array = data_array
            self.data_row_counter = 0
        except Exception as error:
            print("The exception in init_for_granularity_parser of InstanceParse is: " + format(error))
            exit(1)

        self.__isTrain = _isTrain
        self.__attributeCount = 0

    # end of Parser constructor

    #  * It returns all the header read in parseHeader.
    #  * @return a string with the header information.

    def getHeader(self):
        return self.__header

    # end getHeader

    #  * It returns the relation name
    #  * @return a string with the relation name.

    def getRelation(self):
        return self.__relation

    # end getRelation
    #

    #  * It returns an instance
    #  * @return an string with the instance.

    def getInstance(self):
        return self.getLines()

    # end getInstance
    # * It returns the number of attributes
    # * @return an integer with the number of attributes.
    def getAttributeNum(self):
        return self.__attributeCount

    # * This method reads one valid line of the file. So, it ignores the comments,
    # * and empty lines.
    # * @return a string with the new line read.

    def getLines(self):
        try:
            file_first_line = None

            print("In InstanceParser getLines method, the file is " + str(self.file))
            file_strings = self.file.read()
            file_lines = file_strings.splitlines()

            line_Nuember = len(file_lines)
            if line_Nuember != 0:
                print("file has " + str(line_Nuember) + " lines")
            else:
                print("file_lines is empty!!")

            for line in file_lines:
                if (line != "" or line is not None) and not line.startswith("%"):  # line is not empty
                    self.lineCounter = self.lineCounter + 1
                    file_first_line = line

            print("file_lines: " + str(file_lines))
            print("file_first_line: " + str(file_first_line))
            print("In getLines, there are " + str(self.lineCounter) + " lines")

        except Exception as error:
            print("Inside getLines of InstanceParser , Exception is: " + format(error))
            exit(1)

        return file_lines

    # added by rui for granularity rules
    def get_rows(self):

        row_number = len(self.data_row_array)
        if row_number != 0:
            print("row array has " + str(row_number) + " rows")
        else:
            print("row_number is 0 !!")

        self.data_row_counter = row_number
        return self.data_row_array

    # end get_rows

    # This method closes the buffered reader used to parse the instances
    def close(self):
        try:
            print("close file, name is :" + str(self.file.name))
            self.file.close()
        except IOError as ioError:
            print("Error: the instance parser could not be closed. Exiting now." + format(ioError))
            exit(-1)

    # end of Parser class
