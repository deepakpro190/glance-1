from retrieval.search import SearchEngine

engine = SearchEngine()

results = engine.search(

    "red dress",

    top_k=10

)

print()

for r in results:

    print(r)