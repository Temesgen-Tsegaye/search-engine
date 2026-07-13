import os

import numpy as np
from sentence_transformers import SentenceTransformer
class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings=None
        self.documents=None
        self.document_map={}
    def verify_model(model_name: str) -> bool:
        try:
            model= SentenceTransformer(model_name)
            print(f"Model '{model_name}' loaded successfully with max sequence length: {model.max_seq_length}")
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            return False
    
    def add_vectors(vectors1, vectors2):
        if len(vectors1) != len(vectors2):
            raise ValueError("Vectors must have the same length to be added.")
        return [v1 + v2 for v1, v2 in zip(vectors1, vectors2)]
    
    def subtract_vectors(vectors1, vectors2):
        if len(vectors1) != len(vectors2):
            raise ValueError("Vectors must have the same length to be subtracted.")
        return [v1 - v2 for v1, v2 in zip(vectors1, vectors2)]
    
    def dot(vector1,vector2):
        if len(vector1) != len(vector2):
            raise ValueError("Vectors must have the same length to compute dot product.")
        return sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
    
    def generate_embeddings(self, text):
        if text is None or len(text) == 0:
            raise ValueError("Empty string")
        return self.model.encode([text])[0]
    def build_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save("cash/movie_embeddings.npy", self.embeddings)

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists("cash/movie_embeddings.npy"):
            self.embeddings = np.load("cash/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        self.build_embeddings(documents)
        return self.embeddings

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embeddings(query)
        scored = []
        for i, doc in enumerate(self.documents):
            score = cosine_similarity(query_embedding, self.embeddings[i])
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"score": score, "title": doc["title"], "description": doc["description"]}
            for score, doc in scored[:limit]
        ]

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def verify_embeddings():
    import sys
    _cli_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _cli_dir)
    from lib.search_util import load_movies
    searcher = SemanticSearch()
    movies = load_movies()
    embeddings = searcher.load_or_create_embeddings(movies)
    print(f"Number of documents: {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape}")

def embed_query_text(query):
    searcher = SemanticSearch()
    embedding = searcher.generate_embeddings(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")
    return embedding

_model = SemanticSearch()

def embed_text(text):
    return _model.generate_embeddings(text)