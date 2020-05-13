from sklearn.feature_extraction.text import TfidfVectorizer


input_requirement=input("Enter your Search String: ")
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt') # if necessary...
nltk.download('wordnet')

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

hostname = 'localhost'
username = 'root'
password = ''
database = 'code_repository'

# Simple routine to run a query on a database and print the results:
def doQuery( conn ) :
    cur = conn.cursor()
    tb_function_rows=[]
    cur.execute("SELECT ID,FUNCTION_NAME,FUNCTION_BODY,FUNCTION_DESCRIPTOR FROM tb_function ")

    for ID,FUNCTION_NAME,FUNCTION_BODY,FUNCTION_DESCRIPTOR in cur.fetchall() :
        tb_function_rows.append((ID,FUNCTION_NAME,FUNCTION_BODY,FUNCTION_DESCRIPTOR))
    return tb_function_rows

import mysql.connector
myConnection = mysql.connector.connect( host=hostname, user=username, password=password, db=database )
function_summary=doQuery( myConnection )
myConnection.close()


function_summary_with_sim=[]
for s in function_summary:
    similarity=cosine_sim(s[3],input_requirement)
##    print("_______Function__________\n",s[0])
##    print("Similarity With Input:",similarity)
##    print("\n____________________")
    function_summary_with_sim.append((s[2],s[3],similarity))
    


sorted_function_summary_with_sim=sorted(function_summary_with_sim, key=lambda x: x[2], reverse=True)
f= open("output.txt","w+")

for s in sorted_function_summary_with_sim:
    if(s[2]>0):
        f.write("####        Input String Was: "+input_requirement+"\n\n\n")
        f.write("##Function               :\n\n\n")
        f.write(str(s[0]))
        f.write("\n\nSimilarity With Input:")
        f.write(str(s[2]))
        f.write("\n-------------------------------------\n")
f.close()
