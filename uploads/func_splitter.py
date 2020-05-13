import re

import ast
import astor
import glob
from pathlib import Path
from nltk.corpus import wordnet
import re
##import astor
import pandas as pd


from nltk.tokenize import RegexpTokenizer
file_path='test_function_list.py'
with open(file_path) as f:
    lines = f.readlines()

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

def retrieve_comment_from_function(function_name):
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
    
def get_function_summary_pairs(blob):
    "Extract (function/method, docstring) pairs from a given code blob."
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
                    
##            print("variable list start")
##            print(variable_list)
            processed_variable_list=get_processed_variable_list(variable_list)
##            print(processed_variable_list)
##            print("varibale list end")
            docstring = ast.get_docstring(f) if ast.get_docstring(f) else ''
            
            function = source.replace(ast.get_docstring(f, clean=False), '') if docstring else source
##            print("docstring=====",docstring,"end of docstring")
##            print("function=====",function,"end of function")
            comments=retrieve_comment_from_function(f.name)
##            print(retrieve_comment_from_function(f.name))
            pairs.append((f.name,
                          f.lineno,
##                          source,
                          function,
                          docstring.split('\n\n')[0],
                          processed_variable_list,
                          comments
                         ))
    except (AssertionError, MemoryError, SyntaxError, UnicodeEncodeError):
        pass
    return pairs




with open("test_function_list.py", "r") as source:
    pair_list=get_function_summary_pairs(source.read())

function_summary=[]
for element in pair_list:
    summary=''
    summary=element[2]+'\n'
    for variable in element[3]:
        summary+=variable+'\n'
    for comment in element[4]:
        summary+=comment+'\n'
    function_summary.append((element[0],element[2],summary))

    
##    print("function Name=",element[0],"\n")
##    print("Docstring=",element[3],"\n")
##    print("Variables=",element[4],"\n")
##    print("Comments=",element[5],"\n")
##    print("------------------")



input_requirement=input("Enter your Search String: ")

##print(input_requirement)
##
##for s in function_summary:
##    print(s[0],'\n',s[1])

from sklearn.feature_extraction.text import TfidfVectorizer



import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt') # if necessary...


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


function_summary_with_sim=[]
for s in function_summary:
    similarity=cosine_sim(s[2],input_requirement)
##    print("_______Function__________\n",s[0])
##    print("Similarity With Input:",similarity)
##    print("\n____________________")
    function_summary_with_sim.append((s[0],s[1],similarity))
    


sorted_function_summary_with_sim=sorted(function_summary_with_sim, key=lambda x: x[2], reverse=True)
f= open("output.txt","w+")
for s in sorted_function_summary_with_sim:
    f.write("####        Input String Was: "+input_requirement+"\n\n\n")
    f.write("##Function               :\n\n\n")
    f.write(str(s[1]))
    f.write("\n\nSimilarity With Input:")
    f.write(str(s[2]))
    f.write("\n\n\n\n\n\--")
f.close()









