#!/usr/bin/env python3

import argparse

from lib.semantic_search import SemanticSearch, embed_query_text, embed_text, verify_embeddings
from lib.search_util import load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    embed_parser = subparsers.add_parser("embed", help="Generate embedding for text")
    embed_parser.add_argument("text", type=str, help="Text to embed")
    query_parser = subparsers.add_parser("embed_query", help="Generate embedding for a search query")
    query_parser.add_argument("query", type=str, help="Search query to embed")
    search_parser = subparsers.add_parser("search", help="Search movies by semantic similarity")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")
    subparsers.add_parser("verify", help="Verify embeddings by loading or creating them from movies.json")

    args = parser.parse_args()

    match args.command:
        case "embed":
            embedding = embed_text(args.text)
            print(f"Text: {args.text}")
            print(f"First 3 dimensions: {embedding[:3]}")
            print(f"Dimensions: {embedding.shape[0]}")
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            searcher = SemanticSearch()
            movies = load_movies()
            searcher.load_or_create_embeddings(movies)
            results = searcher.search(args.query, args.limit)
            for i, r in enumerate(results, 1):
                desc = r["description"][:80] + "..." if len(r["description"]) > 80 else r["description"]
                print(f'{i}. {r["title"]} (score: {r["score"]:.4f})')
                print(f"  {desc}")
        case "verify":
            verify_embeddings()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

#locality sensitive search   