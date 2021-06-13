import numpy as np
import os
from pathlib import Path
from pandas import DataFrame


class ReadConfigFile:
    lineCounter = 0
    file_first_line = None
    dataset_folder = 'kddcup-rootkit-imap_vs_back'
    config_folder = 'config'
    data_folder = 'dataset'
    config_file = "config0s0.txt"

    file_valid_name = None
    file_train_name = None
    file_test_name = None

    # whole_file_name_with_path = os.path.join(os.getcwd(), config_file)
    cwd = Path.cwd()
    whole_file_name_with_path = cwd / dataset_folder / config_folder / config_file

    dataset_name = None
    attributes_names = []
    classes = []

    def __init__(self):
        print("create a new class object")

    def main(self):
        self.read_config_file(self, self.cwd, self.whole_file_name_with_path)
        self.read_train_data_file(self, self.file_train_name)
        self.read_train_data_file(self, self.file_test_name)

    def read_config_file(self, cwd, whole_file_name_with_path):

        try:
            print("Config File to be opened is : " + str(whole_file_name_with_path))
            with whole_file_name_with_path.open() as data_file:
                print("fileName.open() as data_file......")
                file_lines = data_file.readlines()

            line_number = len(file_lines)
            if line_number != 0:
                print("file has " + str(line_number) + " lines")
            else:
                print("file_lines is empty!!")

            print("file_lines: " + str(file_lines))

            print("In getLines, there are " + str(line_number) + " lines")

        except Exception as error:
            print("Inside read_config_file , Exception is: " + format(error))
            exit(1)

        file_index = -1
        print(cwd)
        for line in file_lines:

            if "inputData" in line:
                sub_line = line.split()
                for each_line in sub_line:
                    # print("each file is :"+str(each_line))
                    if "../" in each_line:
                        file_index = file_index + 1
                        sub_line_two = each_line.split("/")
                        part_file_name = sub_line_two[3]
                        # remove the last "
                        part_file_name = part_file_name[:-1]
                        file_with_path = cwd / self.dataset_folder / self.data_folder / part_file_name
                        if file_index == 0:
                            self.file_valid_name = file_with_path
                            print(" file_valid_name is :" + str(self.file_valid_name))
                        elif file_index == 1:
                            self.file_train_name = file_with_path
                            print(" file_train_name is :" + str(self.file_train_name))
                        elif file_index == 2:
                            self.file_test_name = file_with_path
                            print(" file_test_name is :" + str(self.file_test_name))

    def read_train_data_file(self, file_train_name):
        data_string_array = []
        try:
            print("File to be opened is : " + str(file_train_name))
            with file_train_name.open() as data_file:
                print("fileName.open() as data_file......")
                file_lines = data_file.readlines()

                matrix_column = []
                header = []

                for line in file_lines:
                    if "@" in line:
                        print("@ line is " + str(line))
                        line_string = str(line)
                        if "@relation" in line:
                            sub_line_data = line_string.split()
                            print("sub_line_data")
                            print(sub_line_data)
                            for word in sub_line_data:
                                print("word is :" + str(word))
                            self.dataset_name = sub_line_data[1]
                            print("self.dataset_name is " + self.dataset_name)
                            print("sub_line_data[1] is " + sub_line_data[1])

                        elif "@attribute" in line:
                            sub_line_data = str(line).split()
                            sub_line_data[1] = sub_line_data[1].replace('\'', '')
                            if sub_line_data[1].lower() == "class":
                                header.append("class")
                                for i in range(0, len(sub_line_data)):
                                    print("the i class is :" + sub_line_data[i])
                                    class_name = sub_line_data[i].replace('{', '')
                                    class_name = class_name.replace('}', '')
                                    class_name = class_name.replace(',', '')
                                    class_name = class_name.replace('\n', '')
                                    class_name = class_name.replace('\'', '')
                                    print("class_name:" + class_name)
                                    self.classes.append(class_name)
                            else:
                                print("attributes_names is " + sub_line_data[1])
                                attribute_name = sub_line_data[1]
                                attribute_name = attribute_name.replace('\'', '')
                                print("attributes_names is " + attribute_name)
                                self.attributes_names.append(attribute_name)
                                header.append(attribute_name)



                    else:
                        # print("data line is " + str(line))
                        data_string_array.append(line)
                        datas = line.split(",")
                        row = []
                        for each_date in datas:
                            row.append(each_date)
                        matrix_column.append(row)
                print("the header  is :")
                print(header)
                print("the df shape is :")
                df = DataFrame(matrix_column, columns=header)
                print(df)
                print(df.shape)
                count_result = df.groupby(["class"]).count()
                print(count_result)
                print("value counts of class are  :")
                print(df['class'].value_counts())

        except Exception as error:
            print("Inside read data file , Exception is: " + format(error))

            exit(1)


if __name__ == "__main__":
    read_file = ReadConfigFile
    read_file.main(read_file)
