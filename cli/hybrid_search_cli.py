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
            results = searcher.rrf_search(query, args.k, args.limit)
            for i, r in enumerate(results, 1):
                bm25_rank = r["bm25_rank"] if r["bm25_rank"] is not None else "-"
                semantic_rank = r["semantic_rank"] if r["semantic_rank"] is not None else "-"
                print(f'{i}. {r["title"]}')
                print(f'  RRF Score: {r["score"]:.3f}')
                print(f'  BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}')
                print(f'  {r["document"]}')
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
