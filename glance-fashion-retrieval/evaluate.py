from retrieval.search_engine import SearchEngine


QUERIES = [
    "red tie and white shirt in a formal office",
    "formal business attire inside a modern office",
    "casual weekend outfit for a city walk",
    "blue shirt sitting on a park bench",
]


def run_evaluation():
    engine = SearchEngine()
    results = []

    for query in QUERIES:
        parsed_query = engine.query_parser.parse(query)
        ranked = engine.retrieve(query, top_k=10)
        top = ranked[0] if ranked else None

        overlap = 0
        if top:
            top_categories = {c.lower() for c in top.get("categories", [])}
            query_terms = {t.lower() for t in parsed_query.get("extracted_terms", set())}
            overlap = len(query_terms & top_categories)

        results.append({
            "query": query,
            "top_image_id": top["image_id"] if top else None,
            "top_score": top["final_score"] if top else None,
            "top_components": top.get("component_scores") if top else None,
            "term_overlap": overlap,
            "returned_results": len(ranked),
        })

    print("Sanity-check evaluation (not a full labeled benchmark).")
    for item in results:
        print("=" * 70)
        print(item["query"])
        print("top_image_id:", item["top_image_id"])
        print("top_score:", item["top_score"])
        print("term_overlap:", item["term_overlap"])
        print("returned_results:", item["returned_results"])
        print("top_components:", item["top_components"])

    average_score = sum(item["top_score"] or 0.0 for item in results) / len(results)
    print("average_top_score:", round(average_score, 4))


if __name__ == "__main__":
    run_evaluation()
