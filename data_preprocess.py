import glob
from collections import OrderedDict
import os
import autopep8
import ast
import function_descriptor_pair_generator
# #Database Server Credentials
# hostname = 'localhost'
# username = 'root'
# password = ''
# database = 'code_repository'
#
#
#
# def doQuery( conn ) :
#     cur = conn.cursor()
#     func_list=[]
#     cur.execute("SELECT ID, FUNCTION_NAME from tb_function ")
#
#     for ID, FUNCTION_NAME in cur.fetchall() :
#         func_list.append(FUNCTION_NAME)
#     return func_list

# path = 'data'
output_file_path = 'source_code_raw.txt'
# all_files=[]
# for filename in glob.glob(os.path.join(path,'*.py')):
#     print(filename)
#     all_files.append(filename)
#
#
# print(len(all_files))

path = 'data'

all_files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.py' in file:
            all_files.append(os.path.join(r, file))

print(len(all_files))


def readContentFromFile(file_path):
    with open(file_path) as f:
        code_content = f.read()
        # Check if the content is indented
        # print("indentation and parsing started for:", file_path)
        try:
            indented_code_content=autopep8.fix_code(code_content)
            module = ast.parse(indented_code_content)
        except:
            print("There were some error for the file ",f)
        f.close()
        return indented_code_content
    

def get_pair_list():
    # source_codes=""
    # for file_path in all_files:
    #     source_codes+=readContentFromFile(file_path)
    # print(source_codes)
    # output_file= open("source_code_raw.txt","w+")
    # output_file.write(source_codes)
    # output_file.close()
    with open(output_file_path, "r") as source:
        pair_list = function_descriptor_pair_generator.get_function_summary_pairs(source.read(),output_file_path)
    return pair_list
def get_pair_list_for_file(file_path):
    with open(file_path, "r") as source:
        pair_list = function_descriptor_pair_generator.get_function_summary_pairs(source.read(),output_file_path)
    return pair_list

