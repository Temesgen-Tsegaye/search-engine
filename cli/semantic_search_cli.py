#!/usr/bin/env python3

import argparse
import re
from lib.semantic_search import SemanticSearch, ChunkedSemanticSearch, embed_query_text, embed_text, verify_embeddings
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
    subparsers.add_parser("embed_chunks", help="Build chunked embeddings for all movies")
    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search across chunked embeddings")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of results (default: 5)")
    chunk_parser= subparsers.add_parser("chunk",help="Chunking")
    chunk_parser.add_argument("text",type=str,help="Text to chunk")
    chunk_parser.add_argument(
    "--chunk-size",
    type=int,
    default=200,
    help="Number of words per chunk (default: 200)",
    )
    chunk_parser.add_argument(
    "--overlap",
    type=int,
    default=0,
    help="Number of words per chunk (default: 200)",
    )
    semantic_chunk_parser = subparsers.add_parser(
    "semantic_chunk",
    help="Chunk text semantically by sentences"
)

    semantic_chunk_parser.add_argument(
        "text",
        type=str,
        help="Text to chunk"
    )
    
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum number of sentences per chunk"
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Maximum number of overlap per chunk"
    )
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
        case "embed_chunks":
            searcher = ChunkedSemanticSearch()
            movies = load_movies()
            embeddings = searcher.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            searcher = ChunkedSemanticSearch()
            movies = load_movies()
            searcher.load_or_create_chunk_embeddings(movies)
            results = searcher.search_chunks(args.query, args.limit)
            for i, r in enumerate(results, 1):
                print(f"\n{i}. {r['title']} (score: {r['score']:.4f})")
                print(f"   {r['document']}...")
        case "verify":
            verify_embeddings()
        case "chunk":
         words = args.text.split()

         start = 0

         while start < len(words):
           end = start + args.chunk_size

           chunk_words = words[start:end]

           joined = " ".join(chunk_words)

           print(f"Chunk ({len(joined)} chars):")
           print(joined)
           print()

           if args.overlap > 0:
               start = end - args.overlap
           else:
               start = end
        case "semantic_chunk":
         sentences = re.split(r"(?<=[.!?])\s+", args.text)
     
         start = 0
         chunk_number = 1
     
         print(f"Semantically chunking {len(args.text)} characters")
     
         while start < len(sentences):
             end = start + args.max_chunk_size
     
             chunk_sentences = sentences[start:end]
     
             chunk = " ".join(chunk_sentences)
     
             print(f"{chunk_number}. {chunk}")
     
             chunk_number += 1
     
             if args.overlap > 0:
                 start = end - args.overlap
             else:
                 start = end


        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

#locality sensitive search   