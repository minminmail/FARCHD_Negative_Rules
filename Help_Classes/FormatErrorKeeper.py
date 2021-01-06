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
#  * Created on 28 de enero de 2005, 8:57
#  */

# /**
#  * <p>
#  * <b> FormatErrorKeeper </b>
#  * </p>
#  * This class is a warehouse of format dataset errors. All the errors are stored in this
#  * class, identifying each error by an identifier. At the end of a run, if there has been
#  * some error, an exception is throws, from which the FormatErrorKeeper can be recovered.
#  *
#  * @author Albert Orriols Puig
#  * @version keel0.1
#  */

class FormatErrorKeeper:
    #
    # * A vector where all the errors are stored
    # */

    __errors = []

    # '''
    #  * Creates a new instance of FormatErrorKeeper
    # '''
    def __init__(self):
        self.__errors = []

    # //end FormatErrorKeeper

    # '''
    #  * Adds one error
    #  * @param er is the Error to be added.
    # '''
    def setError(self, err):
        self.__errors.append(err)

    # end setError

    # * Return the information about one error.
    # * @param i is the error that is wanted to be returned.
    # * @return an ErrorInfo object with the error information.

    def getError(self, i):
        return self.__errors[i]

    # end ErrorInfo

    # * Returns the number of errors.
    # * @return an int with the number of errors.

    def getNumErrors(self):
        return len(self.__errors)

    # end getNumErrors
    #
    # '''
    #  * It does return all the errors
    #      * @return all the errors
    # '''
    def getAllErrors(self):
        return self.__errors
    # end getAllErrors

    # '''
    #  * Initializes the error vector
    # '''

    # end init
    # end Class FormatErrorKeeper
