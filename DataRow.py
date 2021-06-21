# store the each row, class results , feature values, label values
# added by rui for negative rule
class DataRow:
    class_value = None
    feature_values = []
    label_values = []
    rule_degree_dic = {}

    def __init__(self, class_value=0, feature_array=None, label_array=None):
        # print("__init__ of data_row")
        # print(" class_value :" + str(class_value))
        # print(" self.class_value :"+str(self.class_value))
        self.class_value = class_value
        self.feature_values = feature_array
        self.label_values = label_array
