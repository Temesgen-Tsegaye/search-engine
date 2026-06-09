
from lib.search_util import load_movies,load_stop_words
import string
from nltk.steamer import PortStemmer
stemmer=PortStemmer()


def clean_text(text:string):
    text=text.lower()
    text=text.translate(str.maketrans("","",string.punctuation))
    return text

def toknize_text(text):
    text=clean_text(text)
    stopwords=load_stop_words()
    tokens=[stemmer.stem(tok) for tok in text.split() if tok and tok not in stopwords]
    return tokens
def has_mathing_token(query_tokens:list[str],title_tokens:list[str]):
    for q_tok in query_tokens:
        for t_tok in title_tokens:
            if q_tok in t_tok:
                return True
    return False


def seach_command(query,n_result):
    data=load_movies()
    res=[]
    query_tokens=toknize_text(query)
    print(f"query tokens: {query_tokens}")
    for movie in data:
            movie_tokens=toknize_text(movie["title"])
            if has_mathing_token(query_tokens,movie_tokens):
                res.append(movie["title"])
                if len(res)>n_result:
                 break
             
        
               
    
    return res

        
