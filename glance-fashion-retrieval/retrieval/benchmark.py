import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from retrieval.search_engine import SearchEngine


QUERIES = [
    {
        "query": "red tie and white shirt in a formal office",
        "relevant_ids": {25910},
    },
    {
        "query": "formal business attire inside a modern office",
        "relevant_ids": {24512},
    },
    {
        "query": "casual weekend outfit for a city walk",
        "relevant_ids": {29681},
    },
    {
        "query": "blue shirt sitting on a park bench",
        "relevant_ids": {26070},
    },
]


def compute_metrics(results, relevant_ids, k=5):
    hits = 0
    reciprocal_rank = 0.0
    ndcg = 0.0

    for rank, item in enumerate(results[:k], start=1):
        if item["image_id"] in relevant_ids:
            hits += 1
            reciprocal_rank = 1.0 / rank
            ndcg += 1.0 / rank

    recall_at_k = hits / max(1, len(relevant_ids))
    mrr = reciprocal_rank
    ndcg_score = ndcg / max(1, len(relevant_ids))
    return recall_at_k, mrr, ndcg_score


def run_benchmark():
    engine = SearchEngine()
    metrics = []

    for item in QUERIES:
        ranked = engine.retrieve(item["query"], top_k=20)
        recall_at_k, mrr, ndcg = compute_metrics(ranked, item["relevant_ids"], k=5)
        metrics.append({
            "query": item["query"],
            "recall_at_5": recall_at_k,
            "mrr": mrr,
            "ndcg@5": ndcg,
        })

    print("Sanity-check benchmark (uses hand-picked relevance IDs and should not be treated as a full accuracy claim).")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run_benchmark()
