import sys
import os
import glob

def main(file_list):
    print("DEBUGGING: file_list = ",file_list)

#------------------------------ MAIN ------------------------------#
if __name__=="__main__":
    print("DEBUGGING: sys.argv = ",sys.argv)
    file_list = [os.path.abspath(el) for el in glob.glob(sys.argv[1])]
    main(file_list)
