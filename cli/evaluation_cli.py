import argparse
import json

from lib.search_util import PROJECT_ROOT, load_movies
from lib.hybrid_search import HybridSearch


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )
    args = parser.parse_args()
    limit = args.limit

    with open(PROJECT_ROOT / "data" / "golden_dataset.json") as f:
        golden = json.load(f)

    documents = load_movies()
    searcher = HybridSearch(documents)

    print(f"k={limit}\n")

    for i, tc in enumerate(golden["test_cases"]):
        query = tc["query"]
        relevant = tc["relevant_docs"]

        results = searcher.rrf_search(query, k=60, limit=limit)
        retrieved_titles = [r["title"] for r in results]

        relevant_retrieved = sum(1 for t in retrieved_titles if t in relevant)
        precision = relevant_retrieved / limit if limit > 0 else 0.0
        recall = relevant_retrieved / len(relevant) if len(relevant) > 0 else 0.0
        f1= 2*(precision*recall)/(precision+recall)

        print(f'- Query: {query}')
        print(f'  - Precision@{limit}: {precision:.4f}')
        print(f'  - Recall@{limit}: {recall:.4f}')
        print(f' --F1@{limit}:{recall:.4f}')
        print(f'  - Retrieved: {", ".join(retrieved_titles)}')
        print(f'  - Relevant: {", ".join(relevant)}')
        if i < len(golden["test_cases"]) - 1:
            print()


if __name__ == "__main__":
    main()
