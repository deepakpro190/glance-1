import json

from indexing.metadata_builder import MetadataBuilder

builder = MetadataBuilder()

with open("artifacts/embeddings/image_ids.json") as f:
    image_ids = json.load(f)

metadata = builder.build(image_ids=image_ids)

builder.save(
    metadata,
    "artifacts/metadata/image_metadata.json"
)