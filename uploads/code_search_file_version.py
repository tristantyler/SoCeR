from sklearn.feature_extraction.text import TfidfVectorizer

import function_descriptor_pair_generator
input_requirement=input("Enter your Search String: ")
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

##nltk.download('punkt') # if necessary...

output_file_path='source_code_raw.txt'
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


function_summary=[]


with open(output_file_path, "r") as source:
    function_summary=function_descriptor_pair_generator.get_function_summary_pairs(source.read(),output_file_path)




function_summary_with_sim=[]
for s in function_summary:
    similarity=cosine_sim(s[2],input_requirement)
##    print("_______Function__________\n",s[0])
##    print("Similarity With Input:",similarity)
##    print("\n____________________")
    function_summary_with_sim.append((s[1],similarity))
    


sorted_function_summary_with_sim=sorted(function_summary_with_sim, key=lambda x: x[1], reverse=True)
f= open("output.txt","w+")

for s in sorted_function_summary_with_sim:
    if(s[1]>0):
        f.write("####        Input String Was: "+input_requirement+"\n\n\n")
        f.write("##Function               :\n\n\n")
        f.write(str(s[0]))
        f.write("\n\nSimilarity With Input:")
        f.write(str(s[1]))
        f.write("\n-------------------------------------\n")
f.close()
