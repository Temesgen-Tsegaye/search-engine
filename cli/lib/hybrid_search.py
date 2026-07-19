import os

from lib.keyword_search import InvertedIndex, toknize_text
from lib.semantic_search import ChunkedSemanticSearch


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = 0.5
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        query_tokens = toknize_text(query)
        scores: dict = {}
        for token in query_tokens:
            for doc_id in self.idx.index.get(token, []):
                scores[doc_id] = scores.get(doc_id, 0) + self.idx.bm25_score(doc_id, token)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [
            {
                "id": self.idx.docmap[doc_id]["id"],
                "title": self.idx.docmap[doc_id]["title"],
                "score": score
            }
            for doc_id, score in ranked
        ]

    def _get_description(self, doc_id: int) -> str:
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc.get("description", "")
        return ""

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, 500 * limit)
        semantic_results = self.semantic_search.search_chunks(query, 500 * limit)

        bm25_scores = [r["score"] for r in bm25_results]
        semantic_scores = [r["score"] for r in semantic_results]

        bm25_norm = normalize_scores(bm25_scores)
        semantic_norm = normalize_scores(semantic_scores)

        combined: dict[int, dict] = {}
        for r, score in zip(bm25_results, bm25_norm):
            doc_id = r["id"]
            combined[doc_id] = {
                "id": doc_id,
                "title": r["title"],
                "document": self._get_description(doc_id),
                "bm25_score": score,
                "semantic_score": 0.0
            }

        for r, score in zip(semantic_results, semantic_norm):
            doc_id = r["id"]
            if doc_id in combined:
                combined[doc_id]["semantic_score"] = score
            else:
                combined[doc_id] = {
                    "id": doc_id,
                    "title": r["title"],
                    "document": self._get_description(doc_id),
                    "bm25_score": 0.0,
                    "semantic_score": score
                }

        for doc_id, data in combined.items():
            data["score"] = hybrid_score(data["bm25_score"], data["semantic_score"], alpha)

        sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, 500 * limit)
        semantic_results = self.semantic_search.search_chunks(query, 500 * limit)

        combined: dict[int, dict] = {}

        for rank, r in enumerate(bm25_results, start=1):
            doc_id = r["id"]
            combined[doc_id] = {
                "id": doc_id,
                "title": r["title"],
                "document": self._get_description(doc_id),
                "bm25_rank": rank,
                "semantic_rank": None,
                "score": rrf_score(rank, k)
            }

        for rank, r in enumerate(semantic_results, start=1):
            doc_id = r["id"]
            if doc_id in combined:
                combined[doc_id]["semantic_rank"] = rank
                combined[doc_id]["score"] += rrf_score(rank, k)
            else:
                combined[doc_id] = {
                    "id": doc_id,
                    "title": r["title"],
                    "document": self._get_description(doc_id),
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "score": rrf_score(rank, k)
                }

        sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:limit]
