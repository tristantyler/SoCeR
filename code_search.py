from sklearn.feature_extraction.text import TfidfVectorizer


# input_requirement=input("Enter your Search String: ")
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
import re

nltk.download('punkt') # if necessary...


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]


def normalize(text):
    '''remove punctuation, lowercase, stem'''
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))


vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')


def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0, 1]


def code_snippet_result(function_summary, search_query):
    function_summary_with_sim = []
    function_summaries = []
    for s in function_summary:
        similarity = cosine_sim(s.function_descriptor, search_query)
        if similarity > 0:
            function_summary_with_sim.append((s.id, s.function_name, s.function_body, similarity))
            for summary in s.function_descriptor.splitlines():
                summary_trimmed = re.sub(' +', ' ', str(summary).lower())
                summary_trimmed = summary_trimmed.strip()
                if len(summary_trimmed.split()) > 1:
                    function_summaries.append(summary_trimmed.strip())
    function_summaries = list(set(function_summaries))
    sorted_function_summary_with_sim = sorted(function_summary_with_sim, key=lambda x: x[3], reverse=True)
    suggested_query_with_sim = []
    for query in function_summaries:
        similarity = cosine_sim(query, search_query)
        if 0 < similarity < 0.99:
            suggested_query_with_sim.append((query, similarity))
        sorted_query_with_sim = sorted(suggested_query_with_sim, key=lambda x: x[1], reverse=True)

    if len(sorted_query_with_sim) > 10:
        sorted_query_with_sim = sorted_query_with_sim[:10]

    # print(search_query)
    output_codes = "####        Input String Was: " + search_query + " \n"

    for s in sorted_function_summary_with_sim:
        if s[3] > 0:
            output_codes += "##Function               :\n"
            output_codes += str(s[2])
            output_codes += "\nSimilarity With Input:"
            output_codes += str(s[3])
            output_codes += "\n----------------------------------------------------------------------\n"
    return output_codes, sorted_query_with_sim
