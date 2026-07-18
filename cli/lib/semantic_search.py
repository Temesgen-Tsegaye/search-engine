import os
import re
import numpy as np
import json
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


SCORE_PRECISION = 4

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks = []
        chunk_metadata = []

        for movie_idx, doc in enumerate(documents):
            description = doc.get("description", "")
            if not description:
                continue

            sentences = re.split(r"(?<=[.!?])\s+", description)

            chunk_size = 4
            overlap = 1

            start = 0
            doc_chunks = []
            while start < len(sentences):
                end = start + chunk_size
                chunk_sentences = sentences[start:end]
                chunk = " ".join(chunk_sentences)
                doc_chunks.append(chunk)

                if overlap > 0:
                    start = end - overlap
                else:
                    start = end

            total_chunks_in_doc = len(doc_chunks)
            for chunk_idx, chunk in enumerate(doc_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": total_chunks_in_doc
                })

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs("cache", exist_ok=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r") as f:
                data = json.load(f)
            self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embeddings(query)

        chunk_scores = []
        for i, chunk_emb in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_emb)
            chunk_scores.append({
                "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
                "score": score
            })

        best_scores = {}
        for cs in chunk_scores:
            movie_idx = cs["movie_idx"]
            if movie_idx not in best_scores or cs["score"] > best_scores[movie_idx]:
                best_scores[movie_idx] = cs["score"]

        sorted_movies = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
        top_movies = sorted_movies[:limit]

        results = []
        for movie_idx, score in top_movies:
            doc = self.documents[movie_idx]
            desc = doc["description"]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": desc,
                "score": round(score, SCORE_PRECISION),
                "metadata": {}
            })

        return results
         

