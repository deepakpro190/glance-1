from retrieval.search_engine import SearchEngine

engine = SearchEngine()

while True:

    print()

    query = input("Search : ")

    if query.lower() == "exit":

        break

    parsed_query = engine.query_parser.parse(query)
    results = engine.retrieve(query)

    print()
    print(f"Parsed terms      : {', '.join(sorted(parsed_query['extracted_terms']))}")

    for rank, item in enumerate(results[:10], start=1):
        print("=" * 70)
        print(f"Rank              : {rank}")
        print(f"Image ID          : {item['image_id']}")
        print(f"Final Score       : {item['final_score']}")
        print(f"Embedding Score   : {item['embedding_score']:.4f}")
        print(f"Category Score    : {item['category_score']:.4f}")
        print(f"Color Score       : {item['color_score']:.4f}")
        print(f"Style Score       : {item['style_score']:.4f}")
        print(f"Environment Score : {item['environment_score']:.4f}")
        print(f"Object Score      : {item['object_score']:.4f}")
        print(f"Categories        : {', '.join(item['categories'])}")
        print(f"Objects           : {item['num_objects']}")
        print(f"Components        : {item['component_scores']}")