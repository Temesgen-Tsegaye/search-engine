
from lib.search_util import BM25_B, BM25_K1, load_movies,load_stop_words
import string
from nltk.stem import PorterStemmer
from collections import Counter, defaultdict
import os
import math
import pickle
from lib.search_util import CASH_PATH
stemmer=PorterStemmer()
class InvertedIndex:
    def __init__(self):
        self.index=defaultdict(set) #  token:[doc_id1,doc_id2,...]
        self.docmap={} #map doc id to document
        self.index_path=CASH_PATH / "index.pkl"
        self.docmap_path=CASH_PATH / "docmap.pkl"
        self.term_frequency=defaultdict(Counter)
        self.term_frequency_path=CASH_PATH / "term_frequency.pkl"
        self.doc_lengths=defaultdict(int)
        self.doc_lengths_path=CASH_PATH / "doc_lengths.pkl"
    def __add_document(self,doc_id:str,text:str):
       tokens=toknize_text(text)
       for tok in set(tokens):
           self.index[tok].add(doc_id)
       self.term_frequency[doc_id].update(tokens)
       self.doc_lengths[doc_id]=len(tokens)
       
    def get_documents(self,term:str)->list[str]:
        return sorted(list(self.index.get(term,[])))    
    def get_tf(self, doc_id: str, term: str) -> int:
        tokens = toknize_text(term)
        if not tokens:
            return 0
        token = tokens[0]
        # Normalize doc_id to int if possible, as movies.json uses int IDs
        normalized_id = int(doc_id) if str(doc_id).isdigit() else doc_id
        return self.term_frequency[normalized_id][token]

    def get_idf(self, term: str) -> float:
        tokens = toknize_text(term)
        if not tokens:
            return 0.0
        token = tokens[0]
        total_docs = len(self.docmap)
        docs_with_term = len(self.index[token])
        if docs_with_term == 0:
            return 0.0
        return math.log((total_docs + 1) / (docs_with_term + 1))

    def get_tfidf(self, doc_id: str, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    # BM25
    def get_bm25_idf(self, term: str) -> float:
        tokens = toknize_text(term)
        if not tokens:
            return 0.0
        token = tokens[0]
        total_docs = len(self.docmap)
        docs_with_term = len(self.index[token])
        if docs_with_term == 0:
            return 0.0
        return math.log((total_docs - docs_with_term + 0.5) / (docs_with_term + 0.5))
    def get_bm25_tf(self, doc_id: str, term: str,k1=BM25_K1,b=BM25_B) -> float:
        term_freq = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)
        return (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * (doc_length / avg_doc_length)))
    
    def __get_avg_doc_length(self) -> float:
        total_length = sum(self.doc_lengths.values())
        total_docs = len(self.docmap)
        return total_length / total_docs if total_docs > 0 else 0.0
    def bm25_score(self, doc_id: str, term: str) -> float:
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term)
        return idf * tf
    def bm25_search(self, query: str, top_k: int = 5) -> list[str]:
        query_tokens = toknize_text(query)
        scores = defaultdict(float)
        for token in query_tokens:
            for doc_id in self.index.get(token, []):
                scores[doc_id] += self.bm25_score(doc_id, token)
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [[self.docmap[doc_id]["title"], score] for doc_id, score in ranked_docs]
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
        with open(self.term_frequency_path, 'wb') as f:
            pickle.dump(self.term_frequency, f)
        with open(self.doc_lengths_path, 'wb') as f:
            pickle.dump(self.doc_lengths, f)
        pass
    def load(self):
        with open(self.index_path, 'rb') as f:
            self.index=pickle.load(f)
        with open(self.docmap_path, 'rb') as f:
            self.docmap=pickle.load(f)
        with open(self.term_frequency_path, 'rb') as f:
            self.term_frequency=pickle.load(f)
        with open(self.doc_lengths_path, 'rb') as f:

            self.doc_lengths=pickle.load(f)

def bm25_search_command(query,n_result):
    idx=InvertedIndex()
    idx.load()
    return idx.bm25_search(query,n_result)            

def bm25_tf_command(doc_id,term):
    idx=InvertedIndex()
    idx.load()
    print(f"BM25 TF for doc_id={doc_id} and term='{term}': {idx.get_bm25_tf(doc_id,term)}")
def bm25_idf_command(term):
    idx=InvertedIndex()
    idx.load()
    print(f"BM25 IDF for term='{term}': {idx.get_bm25_idf(term)}")

def tfidf_command(doc_id,term):
    idx=InvertedIndex()
    idx.load()
    print(f"TF-IDF for doc_id={doc_id} and term='{term}': {idx.get_tfidf(doc_id,term)}")

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

        

def tf_command(doc_id,term):
    idx=InvertedIndex()
    idx.load()
    # Movie ids in the index are stored as integers.
    normalized_doc_id = int(doc_id) if isinstance(doc_id, str) and doc_id.isdigit() else doc_id
    print(idx.get_tf(normalized_doc_id,term))

def idf_command(term):
    idx=InvertedIndex()
    idx.load()
    print(idx.get_idf(term))