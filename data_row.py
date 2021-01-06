# store the each row, class result , feature values, label values
# added by rui for negative rule
class DataRow:
    class_value = None
    feature_values = []
    label_values = []

    def __init__(self):
        self.class_value = 0
        # print("__init__ of data_row")

    def set_three_parameters(self, class_value, feature_array, label_array):
        # print("__init__ of data_row")
        # print(" class_value :" + str(class_value))
        # print(" self.class_value :"+str(self.class_value))
        self.class_value = class_value
        self.feature_values = feature_array
        self.label_values = label_array
