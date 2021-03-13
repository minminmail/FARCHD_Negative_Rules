# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
# This file is part of KEEL - software, the Data Mining tool
# for regression, classification, clustering, pattern  mining and so  on.

# Copyright(C) 2004 - 2010

#     F.Herrera(herrera @ decsai.ugr.es)
#     L.SÃ¡nchez(luciano @ uniovi.es)
#     J.AlcalÃ¡-Fdez(jalcala @ decsai.ugr.es)
#     S.GarcÃ­a(sglopez @ ujaen.es)
#     A.FernÃ¡ndez(alberto.fernandez @ ujaen.es)
#     J.Luengo(julianlm @ decsai.ugr.es)
#
#     This program is free software: you can redistribute it and/or modify
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
#
# ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **
from decimal import Decimal

from Item import Item
from Itemset import Itemset
from RuleBase import RuleBase

import gc
import numpy as np


# /**
#  * <p>Title: Apriori</p>
#  * <p>Description: This class mines the frecuent fuzzy itemsets and the fuzzy classification associacion rules</p>
#  * <p>Copyright: Copyright KEEL (c) 2007</p>
#  * <p>Company: KEEL </p>
#  * @author Written by Jesus Alcala (University of Granada) 09/02/2011
#  * @version 1.0
#  * @since JDK1.6
#  */

class Apriori:
    # ArrayList < Itemset > l2_array;
    l2_array = []
    # double
    minsup = Decimal(0)
    minconf = Decimal(0)

    minSupps_array = []  # double[]
    # int
    nclasses = None
    nvariables = None
    depth = None

    rule_stage1 = 0  # long
    rule_base = None  # RuleBase
    rule_base_class = None  # RuleBase
    train = None  # myDataset
    data_base = None  # DataBase

    #  **
    #  *Default  Constructor.
    #  *

    def __init__(self):
        pass

    # / **
    # * Builder
    # * @ param ruleBase Rule base
    # * @ param dataBase Data Base
    # * @ param train Training dataset
    # * @ param minsup Minimum support.
    # * @ param minconf Maximum Confidence.
    # * @ param depth Depth of the trees (Depthmax)
    # * /

    def multiple_init(self, rule_base_pass, data_base_pass, train, minsup, minconf, depth):
        self.train = train
        self.data_base = data_base_pass
        self.rule_base = rule_base_pass
        self.minconf = minconf
        self.depth = depth
        self.nclasses = self.train.get_nclasses()
        self.nvariables = self.train.get_ninputs()

        self.l2_array = []

        self.minSupps_array = np.empty(self.nclasses, dtype=float)

        for i in range(0, self.nclasses):
            self.minSupps_array[i] = Decimal(self.train.get_frequent_class(i) * minsup)

    # / **
    # * Generate the rule set (Stage 1 and 2)
    # * /

    def generate_rb(self):
        self.rule_stage1 = 0
        # int

        self.rule_base_class = RuleBase()
        self.rule_base_class.init_with_five_parameters(self.data_base, self.train, self.rule_base.get_k_value(),
                                                       self.rule_base.get_inference_type())

        for i in range(0, self.nclasses):
            self.minsup = Decimal(self.minSupps_array[i])
            self.generate_l2_array(i)
            self.generate_large(self.l2_array, i)

            self.rule_base_class.reduce_rules(i)

            self.rule_base.add_rule_base(self.rule_base_class)
            self.rule_base_class.clear()
            gc.collect()

    # each vairable, with different label, each possible class value
    def generate_l2_array(self, class_pass):

        if len(self.l2_array) != 0:
            self.l2_array.clear()
        itemset = Itemset(class_pass)

        for i in range(0, self.nvariables):
            if self.data_base.num_labels(i) > 1:
                for j in range(0, self.data_base.num_labels(i)):
                    item = Item(i, j)
                    itemset.add(item)
                    itemset.calculate_supports(self.data_base, self.train)
                    if itemset.get_support_class() >= self.minsup:
                        self.l2_array.append(itemset.clone())
                    itemset.remove(0)

        self.generate_rules(self.l2_array, class_pass)

        '''
         * Indentifies how many times a class has been uncovered.
         * @param clas Class given to compute the number of times.
         * @return number of times that class has been uncovered.
        '''

    def has_uncover_class(self, class_pass):
        # int
        uncover = None
        degree = Decimal(0)
        itemset = None
        stop = None

        uncover = 0
        for j in range(0, self.train.size()):
            if self.train.getOutputAsInteger(j) == class_pass:
                stop = False
                for i in range(0, len(self.l2_array)):
                    if not stop:
                        itemset = self.l2_array[i]
                        degree = itemset.degree(self.data_base, self.train.getExample(j))
                        if degree > 0.0:
                            stop = True

                if not stop:
                    uncover = uncover + 1

        return uncover
    
    #
    def generate_large(self, Lk, class_pass):
        # int
        i = None
        j = None
        size = None
        l_new = []
        new_itemset = None
        itemseti = None
        itemsetj = None

        size = len(Lk)

        if size > 1:
            if (Lk[0].size() < self.nvariables) and Lk[0].size() < self.depth:
                l_new = []
                for i in range(0, size - 1):
                    itemseti = Lk[i]
                    for j in range(i + 1, size):
                        itemsetj = Lk[j]
                        if self.is_combinable(itemseti, itemsetj):
                            new_itemset = itemseti.clone()
                            new_itemset.add((itemsetj.get(itemsetj.size() - 1)).clone())
                            new_itemset.calculate_supports(self.data_base, self.train)
                            if new_itemset.get_support_class() >= self.minsup:
                                l_new.append(new_itemset)

                    l_new = self.generate_rules(l_new, class_pass)
                    self.generate_large(l_new, class_pass)
                    l_new.clear()
               
                    gc.collect()

    def is_combinable(self, itemseti, itemsetj):
        # int
        i = None
        itemi = None
        itemj = None
        itemset = None

        itemi = itemseti.get(itemseti.size() - 1)
        itemj = itemsetj.get(itemseti.size() - 1)
        if itemi.get_variable() >= itemj.get_variable():
            return False

        return True

    # /**
    #  * Returns the rules generated on the Stage 1.
    #  * @return the rules of the Stage 1
    #  */
    def get_rules_stage1(self):
        return self.rule_stage1

    #generate confident >0.4 rules
    def generate_rules(self, lk, class_pass):
        # int
        i = None
        uncover = None
        itemset = None
        confidence = Decimal(0)
        for i in range(len(lk) - 1, -1, -1):
            itemset = lk[i]

            if itemset.get_support() > 0.0:
                confidence = itemset.get_support_class() / itemset.get_support()
            else:
                confidence = 0.0
            if confidence > 0.4:
                self.rule_base_class.add_itemset(itemset)
                self.rule_stage1 = self.rule_stage1 + 1
            if confidence > self.minconf:
                lk.pop(i)
                #lk = np.delete(lk, i)

        if self.rule_base_class.get_size() > 500000:
            self.rule_base_class.reduce_rules(class_pass)
            gc.collect()

        return lk
