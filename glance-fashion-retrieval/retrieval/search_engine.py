import json
import faiss
import numpy as np
import torch

from transformers import AutoModel, AutoProcessor
from retrieval.query_parser import QueryParser
from retrieval.reranker import FashionReranker
from retrieval.learned_reranker import LearnedReranker
from retrieval.bound_pair_reranker import BoundPairReranker
from config import (
    MODEL_NAME,
    DEVICE,
    FAISS_INDEX_PATH,
    IMAGE_IDS_PATH,
    METADATA_PATH,
    TOP_K
)


class SearchEngine:

    def __init__(self):
        self.query_parser = QueryParser()
        self.reranker = FashionReranker()
        self.bound_pair_reranker = BoundPairReranker()

        print("=" * 60)
        print("Initializing Search Engine")
        print("=" * 60)

        # ---------------------------------------------------
        # Load SigLIP
        # ---------------------------------------------------

        print("Loading SigLIP...")

        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)

        self.model = AutoModel.from_pretrained(MODEL_NAME)

        self.model.to(DEVICE)

        self.model.eval()

        print("SigLIP Loaded.")

        # ---------------------------------------------------
        # Load FAISS
        # ---------------------------------------------------

        print("Loading FAISS Index...")

        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        print(f"Indexed vectors : {self.index.ntotal}")

        # ---------------------------------------------------
        # Load Image IDs
        # ---------------------------------------------------

        with open(IMAGE_IDS_PATH) as f:

            self.image_ids = json.load(f)

        # ---------------------------------------------------
        # Load Metadata
        # ---------------------------------------------------

        with open(METADATA_PATH) as f:

            self.metadata = json.load(f)

        print("Metadata Loaded.")

        self.learned_reranker = LearnedReranker()
        if self.learned_reranker.load():
            print("Loaded learned reranker model.")
        else:
            print("Training learned reranker model...")
            self.learned_reranker.fit(self.metadata)

        print("=" * 60)

    #########################################################

    def encode_query(self, query):

        inputs = self.processor(
            text=[query],
            return_tensors="pt",
            padding=True
        )

        inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model.get_text_features(**inputs)

            # Newer transformers
            if hasattr(outputs, "pooler_output"):
                features = outputs.pooler_output
            else:
                features = outputs

            features = features / features.norm(
                dim=-1,
                keepdim=True
            )

        return features.cpu().numpy().astype(np.float32)
    #########################################################

    def retrieve(self, query, top_k=TOP_K):

        query_embedding = self.encode_query(query)

        scores, indices = self.index.search(

            query_embedding,

            top_k

        )

        candidates = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            image_id = int(self.image_ids[idx])
            meta_lookup = None
            for meta_item in self.metadata.values():
                if int(meta_item.get("image_id", -1)) == image_id:
                    meta_lookup = meta_item
                    break
            if meta_lookup is None:
                continue
            meta = meta_lookup

            candidates.append({
                "image_id": image_id,
                "embedding_score": float(score),
                "categories": meta.get("categories", []),
                "category_ids": meta.get("category_ids", []),
                "colors": meta.get("colors", []),
                "styles": meta.get("styles", []),
                "environments": meta.get("environments", []),
                "garment_types": meta.get("garment_types", []),
                "garment_pairs": meta.get("garment_pairs", []),
                "garment_attributes": meta.get("garment_attributes", []),
                "num_objects": meta.get("num_objects", 0),
                "width": meta.get("width", 0),
                "height": meta.get("height", 0),
            })

        parsed_query = self.query_parser.parse(query)

        results = self.reranker.rerank(
            parsed_query,
            candidates
        )

        bound_results = self.bound_pair_reranker.rerank(parsed_query, results)

        for item in bound_results:
            item["learned_score"] = self.learned_reranker.score(item, parsed_query)
            item["composite_score"] = round(
                0.55 * item.get("raw_final_score", item["final_score"]) + 0.25 * item["learned_score"] + 0.20 * item.get("bound_pair_score", 0.0),
                4,
            )
            item["final_score"] = item.get("final_score", 0.0)

        bound_results.sort(key=lambda x: x["final_score"], reverse=True)
        return bound_results