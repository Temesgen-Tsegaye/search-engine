
from lib.search_util import load_movies,load_stop_words
import string
from nltk.stem import PorterStemmer
from collections import defaultdict
import os
import pickle
from lib.search_util import CASH_PATH
stemmer=PorterStemmer()


class InvertedIndex:
    def __init__(self):
        self.index=defaultdict(set) #  token:[doc_id1,doc_id2,...]
        self.docmap={} #map doc id to document
        self.index_path=CASH_PATH / "index.pkl"
        self.docmap_path=CASH_PATH / "docmap.pkl"
    def __add_document(self,doc_id:str,text:str):
       tokens=toknize_text(text)
       for tok in set(tokens):
           self.index[tok].add(doc_id)
       
    def get_documents(self,term:str)->list[str]:
        return sorted(list(self.index.get(term,[])))    
        
    def build(self):
        movies=load_movies()
        for movie in movies:
            id=movie["id"]
            text=f"{movie['title']} {movie['description']}"
            self.__add_document(id, text)
            self.docmap[id]=movie
        pass
    def save(self):
        os.makedirs(CASH_PATH, exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
        pass
    def load(self):
        with open(self.index_path, 'rb') as f:
            self.index=pickle.load(f)
        with open(self.docmap_path, 'rb') as f:
            self.docmap=pickle.load(f)


def build_index():
    docs=InvertedIndex()
    docs.build()
    docs.save()
    docs.get_documents("merdia")
    print(f"first document with token media: {docs.get_documents('merida')[0]}")



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
    idx=InvertedIndex()
    idx.load()
    seen,res= set(),[]
    query_tokens=toknize_text(query)
    for qt in query_tokens:
        matching_doc_ids=idx.get_documents(qt)
        for matching_doc_id in matching_doc_ids:
            if matching_doc_id in seen:
                continue
            seen.add(matching_doc_id)
            res.append(idx.docmap[matching_doc_id]["title"])
            if len(res)>=n_result:
                return res
        return res
       
            
                

              
           

             
        
               
    
    return res

        
