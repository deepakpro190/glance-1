from retrieval.search_engine import SearchEngine

engine = SearchEngine()

results = engine.retrieve(

    "red dress",

    top_k=10

)

print()

for r in results:

    print(r)