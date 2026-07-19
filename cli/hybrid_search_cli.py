#!/usr/bin/env python3

import argparse

from lib.search_util import load_movies


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    weighted_parser = subparsers.add_parser("weighted-search", help="Weighted hybrid search")
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for semantic score (0-1)")
    weighted_parser.add_argument("--limit", type=int, default=5, help="Number of results")

    rrf_parser = subparsers.add_parser("rrf-search", help="Reciprocal Rank Fusion hybrid search")
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument("-k", type=int, default=60, help="RRF constant")
    rrf_parser.add_argument("--limit", type=int, default=5, help="Number of results")
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Re-ranking method",
    )

    normalize_parser = subparsers.add_parser("normalize", help="Min-max normalize scores")
    normalize_parser.add_argument("scores", type=float, nargs="*", help="Scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "weighted-search":
            from lib.hybrid_search import HybridSearch
            searcher = HybridSearch(load_movies())
            results = searcher.weighted_search(args.query, args.alpha, args.limit)
            for i, r in enumerate(results, 1):
                print(f'{i}. {r["title"]}')
                print(f'  Hybrid Score: {r["score"]:.3f}')
                print(f'  BM25: {r["bm25_score"]:.3f}, Semantic: {r["semantic_score"]:.3f}')
                print(f'  {r["document"]}')
        case "rrf-search":
            import time
            from lib.hybrid_search import HybridSearch
            query = args.query
            if args.enhance:
                from lib.llm import correct_spelling, expand_query
                if args.enhance == "spell":
                    enhanced = correct_spelling(query)
                elif args.enhance == "expand":
                    enhanced = expand_query(query)
                else:
                    enhanced = query
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced}'\n")
                query = enhanced
            searcher = HybridSearch(load_movies())
            rerank_limit = args.limit
            if args.rerank_method:
                rerank_limit *= 5
            results = searcher.rrf_search(query, args.k, rerank_limit)
            if args.rerank_method == "individual":
                from lib.llm import rerank_individual
                print(f"Re-ranking top {len(results)} results using individual method...")
                for r in results:
                    score = rerank_individual(query, r)
                    r["rerank_score"] = score
                    time.sleep(3)
                results.sort(key=lambda x: x["rerank_score"], reverse=True)
            elif args.rerank_method == "batch":
                from lib.llm import rerank_batch
                print(f"Re-ranking top {len(results)} results using batch method...")
                ranked_ids = rerank_batch(query, results)
                rank_map = {doc_id: rank for rank, doc_id in enumerate(ranked_ids, start=1)}
                for r in results:
                    r["rerank_rank"] = rank_map.get(r["id"], len(results))
                results.sort(key=lambda x: x["rerank_rank"])
            elif args.rerank_method == "cross_encoder":
                from sentence_transformers import CrossEncoder
                print(f"Re-ranking top {len(results)} results using cross_encoder method...")
                pairs = [[query, f"{r['title']} - {r['document']}"] for r in results]
                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2", device="cpu")
                scores = cross_encoder.predict(pairs)
                for r, score in zip(results, scores):
                    r["cross_encoder_score"] = score
                results.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
            print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
            for i, r in enumerate(results[:args.limit], 1):
                bm25_rank = r["bm25_rank"] if r["bm25_rank"] is not None else "-"
                semantic_rank = r["semantic_rank"] if r["semantic_rank"] is not None else "-"
                print(f'{i}. {r["title"]}')
                if args.rerank_method == "individual":
                    print(f'   Re-rank Score: {r.get("rerank_score", 0):.3f}/10')
                elif args.rerank_method == "batch":
                    print(f'   Re-rank Rank: {r.get("rerank_rank", "-")}')
                elif args.rerank_method == "cross_encoder":
                    print(f'   Cross Encoder Score: {r.get("cross_encoder_score", 0):.3f}')
                print(f'   RRF Score: {r["score"]:.3f}')
                print(f'   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}')
                print(f'   {r["document"]}')
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
