from retrieval.query_parser import QueryParser

parser = QueryParser()

queries = [

    "White formal dress with black shoes",

    "Blue shirt in office",

    "Casual red jacket",

    "Green skirt for wedding",

    "Business suit"

]

for q in queries:

    print("=" * 60)

    print(q)

    print()

    print(parser.parse(q))