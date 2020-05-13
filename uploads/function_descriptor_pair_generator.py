import re

import ast
import astor
import glob
from pathlib import Path
from nltk.corpus import wordnet
import re
import io
##import astor
import pandas as pd

nltk.download('wordnet')
from nltk.tokenize import RegexpTokenizer


##file_path='source_code_raw.txt'
def get_processed_variable_list(variable_list):
    processed_variable_list=[]
    under_score="_"
    for variable in variable_list:
        if(under_score in variable):
##            print(variable.replace("_"," "))
            processed_variable_list.append(variable.replace("_"," "))
        else:
            splitted = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', variable)).split()
            if(len(splitted)>1):
                variable_phrase=""
                for i in splitted:
                    variable_phrase+=" "+i
                processed_variable_list.append(variable_phrase)
            elif(wordnet.synsets(splitted[0]) and len(splitted[0])>1):
                processed_variable_list.append(splitted[0])
            
    return processed_variable_list



def get_comments(func_body):
    single_line_comments=[]
    for line in func_body.splitlines():
        i = line.rfind('#')
        if i >= 0:
            line = line[i+1:]
            single_line_comments.append(line.strip())
    return single_line_comments

##    print('single line comments',single_line_comments)

def retrieve_comment_from_function(function_name,file_path):
    code_snippet=""
    if (function_name!=''):
        full_function_name = 'def ' + function_name
        function_body = ''
        with open(file_path) as f:
            lines = f.readlines()

        inside_function = 0
        count = 0

        for x in lines:
            count += 1
            if (x.find(full_function_name) > -1):
                function_body += x
                inside_function = 1
                break
        for x in range(count, len(lines)):
            if (len(lines[x]) - len(lines[x].lstrip()) > 0):
                function_body += lines[x]
            else:
                break
        code_snippet+=function_body
    
        f.close()
    comments=get_comments(code_snippet)
    return comments


def get_function_summary_pairs(blob,file_path):
    "Extract (function/method, docstring) pairs from a given code blob."
    function_info = []
    pairs = []
    try:
        module = ast.parse(blob)
        classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
        functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
        for _class in classes:
            functions.extend([node for node in _class.body if isinstance(node, ast.FunctionDef)])

        for f in functions:
            variable_list=[]
            processed_variable_list=[]
            source = astor.to_source(f)
            function_ast=ast.parse(source)
            for func_node in ast.walk(function_ast):
                if isinstance(func_node, ast.Name) and isinstance(func_node.ctx, ast.Store):
                    variable_list.append(func_node.id)
                elif isinstance(func_node, ast.Attribute):
                    variable_list.append(func_node.attr)
                elif isinstance(func_node, ast.FunctionDef):
                    variable_list.append(func_node.name)
                    

            processed_variable_list=get_processed_variable_list(variable_list)
            docstring = ast.get_docstring(f) if ast.get_docstring(f) else ''
            
##            print(docstring)
            function = source.replace(ast.get_docstring(f, clean=False), '') if docstring else source
            comments=retrieve_comment_from_function(f.name,file_path)
            function_info.append((f.name,
                          f.lineno,
##                          source,
                          function,
                          docstring,
                          processed_variable_list,
                          comments
                         ))
           
        for element in function_info:
            summary=''
##            summary=element[0]+'\n'
            for docstr in element[3]:
                summary+=docstr
            for variable in element[4]:
                summary+=variable+'\n'
            for comment in element[5]:
                summary+=comment+'\n'
##            print("***\n",summary,"****")
            pairs.append((element[0],element[2],summary)) ## function name, body, summary
    except (AssertionError, MemoryError, SyntaxError, UnicodeEncodeError):
        pass
    return pairs











