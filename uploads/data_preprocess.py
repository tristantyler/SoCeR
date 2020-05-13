import glob
from collections import OrderedDict
import os
import autopep8
import ast
import function_descriptor_pair_generator
#Database Server Credentials
hostname = 'localhost'
username = 'root'
password = ''
database = 'code_repository'

def doQuery( conn ) :
    cur = conn.cursor()
    func_list=[]
    cur.execute("SELECT ID, FUNCTION_NAME from tb_function ")

    for ID, FUNCTION_NAME in cur.fetchall() :
        func_list.append(FUNCTION_NAME)
    return func_list





path = 'data'
output_file_path='source_code_raw.txt'
all_files=[]
for filename in glob.glob(os.path.join(path,'*.py')):
    print(filename)
    all_files.append(filename)


print(all_files)




def readContentFromFile(file_path):
    with open(file_path) as f:
        code_content = f.read()
        ## Check if the content is indented
##        print("indentation and parsing started for:", file_path)
        try:
            indented_code_content=autopep8.fix_code(code_content)
            module =ast.parse(indented_code_content)
        except:
            print("There were some error for the file ",f)
        f.close()
        return indented_code_content
    


def write_codes_in_file():
    source_codes=""    
    for file_path in all_files:
        source_codes+=readContentFromFile(file_path)

    output_file= open("source_code_raw.txt","w+")
    output_file.write(source_codes)
    output_file.close()

write_codes_in_file()


##with open(output_file_path, "r") as source:
##    pair_list=function_descriptor_pair_generator.get_function_summary_pairs(source.read(),output_file_path)




