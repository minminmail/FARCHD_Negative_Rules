
from os import listdir
from os.path import isfile,join
from Main import Main
import sys

class MultiMain:


    if __name__=='__main__':
        # print("sys.argv: " + sys.argv[1])
        config_files_folder = sys.argv[1]
        con_files = listdir(config_files_folder)
        loopNumber = 1
        for file in con_files:
            if file != "dataset":
                whole_path_file = join(config_files_folder, file)
                main = Main()
                main.executeMultiFiles(whole_path_file)
                # print("finished loop " +str(loopNumber))
                loopNumber =loopNumber + 1