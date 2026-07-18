from retrieval.query_parser import QueryParser

parser = QueryParser()

queries = [

    "white dress",

    "red tie",

    "formal blue shirt",

    "green skirt"

]

for q in queries:

    print("="*60)

    print(q)

    print(parser.parse(q))