from indexing.faiss_builder import FAISSBuilder

builder = FAISSBuilder()

builder.build(
    "artifacts/embeddings/image_embeddings.npy"
)

builder.save(
    "artifacts/faiss/image_index.faiss"
)