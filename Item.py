# /**
#  * <p>Title: Item</p>
#  *
#  * <p>Description: This class contains the representation of a item</p>
#  *
#  * <p>Copyright: Copyright KEEL (c) 2007</p>
#  *
#  * <p>Company: KEEL </p>
#  *
#  * @author Jesus AlcalÃ¡ (University of Granada) 09/02/2011
#  * @version 1.0
#  * @since JDK1.6
#  */


class Item:
    # int
    variable = None
    label = None

    # /**
    #  * Default constructor.
    #  * None attribute will be initialized.
    #  */
    def __init__(self, variable, label):
        self.variable = variable
        self.label = label

    # * It sets the pair of values to the item
    # * </p>
    # * @param variable Value which represents an input attribute of a rule
    # * @param value Value attached to the variable

    def set_values(self, variable, label):
        self.variable = variable
        self.label = label

    # * It returns the variable of the item
    # * </p>
    # * @return Input attribute

    def get_variable(self):
        return self.variable

    # * It returns the label of the item
    # * @return label of the item
    def get_label(self):
        return self.label

    def clone(self):
        d = Item(self.variable, self.label)
        return d

    #   * Function to check if an item is equal to another given
    #   * @param a Item to compare with ours
    #   * @return boolean true = they are equal, false = they aren't.
    def is_equal(self, a_item):
        if (self.variable == a_item.variable) and (self.label == a_item.label):
            return True
        else:
            return False

    # * Function to compare objects of the Item class.
    # * Necessary to be able to use "sort" function.
    # * It sorts in an decreasing order of attribute.
    # * If equals, in an decreasing order of attribute's label.
    # * @param a Item object to compare with.
    # * @return -1 if a is bigger, 1 if smaller and 0 otherwise.

    def compare_to(self, a_object):
        if a_object.variable > self.variable:
            return -1

        elif a_object.variable < self.variable:
            return 1
        elif a_object.label > self.label:
            return -1

        elif a_object.label < self.label:
            return 1

        return 0
