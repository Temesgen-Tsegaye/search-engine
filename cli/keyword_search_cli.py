#!/usr/bin/env python3

import argparse
from lib.keyword_search import idf_command, seach_command,build_index,tf_command, tfidf_command
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    subparsers.add_parser("build", help="Build and save the inverted index cache")
    tf_parser = subparsers.add_parser("tf", help="Calculate term frequency for a given document and term")
    search_parser.add_argument("query", type=str, help="Search query")
    tfidf_parser=subparsers.add_parser("tfidf", help="Calculate TF-IDF for a given document and term")
    tfidf_parser.add_argument("doc_id", type=str, help="Document ID for TF-IDF calculation")
    tfidf_parser.add_argument("term", type=str, help="Term to calculate TF-IDF")
    tf_parser.add_argument("doc_id", type=str, help="Document ID for term frequency")
    tf_parser.add_argument("term", type=str, help="Term to calculate term frequency")
    idf_parser = subparsers.add_parser("idf", help="Calculate inverse document frequency for a given term")
    idf_parser.add_argument("term", type=str, help="Term to calculate inverse document frequency")

    args = parser.parse_args()

    match args.command:
        case "search":
             # print the search query here
             result=seach_command(args.query,5)
             for item in result:
                 print(item)
        case "build":
            build_index()
        case "tf":
            tf_command(args.doc_id,args.term)
        case "idf":
            idf_command(args.term)
        case "tfidf":
            tfidf_command(args.doc_id,args.term)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()