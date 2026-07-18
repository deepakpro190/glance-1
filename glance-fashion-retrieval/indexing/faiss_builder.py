import faiss
import numpy as np
import os


class FAISSBuilder:

    def __init__(self):
        self.index = None

    def build(self, embedding_path):

        print("Loading embeddings...")

        embeddings = np.load(embedding_path).astype(np.float32)

        dimension = embeddings.shape[1]

        print(f"Embeddings : {embeddings.shape}")

        print(f"Embedding Dimension : {dimension}")

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        print(f"Indexed {self.index.ntotal} vectors.")

    def save(self, output_path):

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        faiss.write_index(self.index, output_path)

        print(f"\nFAISS Index Saved -> {output_path}")