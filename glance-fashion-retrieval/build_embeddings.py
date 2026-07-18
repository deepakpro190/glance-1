import json
import os
import numpy as np

from tqdm import tqdm

from indexing.parser import FashionpediaParser
from indexing.embedder import SigLIPEmbedder


OUTPUT_DIR = "artifacts/embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = FashionpediaParser()

embedder = SigLIPEmbedder()

embeddings = []

image_ids = []

BATCH_SIZE = 32

TOTAL_IMAGES = len(parser.train)

print(f"\nGenerating embeddings for {TOTAL_IMAGES} images...\n")

for start in tqdm(
    range(0, TOTAL_IMAGES, BATCH_SIZE),
    desc="Generating Embeddings"
):

    images = []

    ids = []

    end = min(start + BATCH_SIZE, TOTAL_IMAGES)

    for i in range(start, end):

        sample = parser.get_sample(i)

        images.append(sample.image)

        ids.append(sample.image_id)

    batch_embeddings = embedder.embed_batch(images)

    embeddings.extend(batch_embeddings)

    image_ids.extend(ids)

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "image_embeddings.npy"
    ),
    embeddings
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "image_ids.json"
    ),
    "w"
) as f:

    json.dump(image_ids, f)

print("\nEmbeddings Saved Successfully!")

print("Embedding Shape :", embeddings.shape)