from __future__ import division

# /**
#  * <p>Title: Itemset</p>
#  * <p>Description: This class contains the representation of a itemset</p>
#  * <p>Copyright: Copyright KEEL (c) 2007</p>
#  * <p>Company: KEEL </p>
#  * @version 1.0
#  * @since JDK1.6
#  */
from decimal import Decimal

from Item import Item
from MyDataSet import MyDataSet
from DataBase import DataBase


class Itemset:
    itemset = []
    # int
    class_value = None
    # double
    support = None
    support_rule = None

    # /**
    #  * Default constructor.
    #
    #  * Builder
    #  * @param class_value_pass Class
    #  */
    def __init__(self, class_value_pass):
        # print("Itemset init ....")
        self.itemset = []
        self.class_value = class_value_pass
        self.support = 0
        self.support_rule = 0

    # * Clone
    # * @return Return a copy of the itemset
    def clone(self):
        d_itemset = Itemset(self.class_value)
        for i in range(0, len(self.itemset)):
            d_itemset.add((self.itemset[i]).clone())
            d_itemset.class_value = self.class_value
            d_itemset.support = self.support
            d_itemset.support_rule = self.support_rule
        return d_itemset

    # * Function to add an item to our itemset
    # * @param item Element to be added

    def add(self, item):
        self.itemset.append(item)

    # * It returns the item located in the given position of the itemset
    # * @param pos Position of the requested item into the itemset
    # * @return The requested item of the itemset

    def get(self, pos):
        return self.itemset[pos]

    #  /**
    # * Function to remove the item located in the given position
    # * </p>
    # * @param pos Position of the requested item into the itemset
    # * @return The removed item of the itemset

    def remove(self, pos):
        return self.itemset.pop(pos)

    # * It returns the size of the itemset (the number of items it has)
    # * @return Number of items the itemset stores

    def size(self):
        return len(self.itemset)

    # /**
    #  * <p>
    #  * It returns the support of the antecedent of the itemset
    #  * </p>
    #  * @return Support of the antecedent of the itemset
    #  */
    def get_support(self):
        return self.support

    #   * It returns the support of the itemset for its related output class
    #   * @return Support of the itemset for its related output class

    def get_support_class(self):
        return self.support_rule

    # * It returns the output class of the itemset
    # * @return output class of the itemset
    def get_class(self):
        return self.class_value

    # /**
    #  * Set the class with the value given as argument.
    #  * @param clas class given.
    #  */
    def set_class(self, class_pass):
        self.class_value = class_pass

    # * Function to check if an itemset is equal to another given
    # * @param a Itemset to compare with ours
    # * @return boolean true = they are equal, false = they aren't.

    def is_equal(self, a_itemset):
        i = None
        item = None

        if len(self.itemset) != len(a_itemset):
            return False
        if self.class_value != a_itemset.get_class():
            return False
        for i in range(0, len(self.itemset)):
            self.item = self.itemset[i]
            if not self.item.isEqual(a_itemset[i]):
                return False
        return True

    """  
    It computes the support, rule support, hits, misses and PER of our itemset for a given dataset
    
    @param dataBase Given training dataset useful information to calculate supports.
    @param train Given training dataset to be able to calculate supports.
    """

    def calculate_supports(self, dataBase, train):

        degree =0.0
        self.support = 0.0
        self.support_rule =0.0
        for i in range(0, train.size()):
            degree = self.degree_product(dataBase, train.get_example(i))
            self.support = self.support + degree
            if train.get_output_as_integer_with_pos(i) == self.class_value:
                self.support_rule = self.support_rule + degree
        train_data_number = train.get_ndata()
        if train_data_number is 0 or None:
            self.support = 0
            self.support_rule = 0
        else:
            self.support = self.support / train_data_number
            self.support_rule = self.support_rule / train_data_number

    def degree_product(self, dataBase, example):

        degree = 1.0
        for i in range(0, len(self.itemset)):
            if degree > 0.0:
                item = self.itemset[i]
                degree *= dataBase.matching(item.get_variable(), item.get_label(), example[item.get_variable()])
        return degree
