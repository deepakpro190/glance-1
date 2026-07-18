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

        self.metadata_by_image_id = {
            int(item.get("image_id")): item
            for item in self.metadata.values()
            if item.get("image_id") is not None
        }

        self.last_retrieval_meta = {
            "mode": "default",
            "message": "",
        }

        print("Metadata Loaded.")

        self.learned_reranker = LearnedReranker()
        if self.learned_reranker.load():
            print("Loaded learned reranker model.")
        else:
            print("Training learned reranker model...")
            self.learned_reranker.fit(self.metadata)

        print("=" * 60)

    #########################################################

    def _has_any_constraints(self, parsed_query):
        return any([
            bool(parsed_query.get("categories", [])),
            bool(parsed_query.get("colors", [])),
            bool(parsed_query.get("styles", [])),
            bool(parsed_query.get("environments", [])),
            bool(parsed_query.get("attribute_bindings", [])),
        ])

    #########################################################

    def _candidate_matches(self, candidate, parsed_query, strict=True):
        query_categories = set(parsed_query.get("categories", []))
        query_colors = set(parsed_query.get("colors", []))
        query_styles = set(parsed_query.get("styles", []))
        query_environments = set(parsed_query.get("environments", []))
        bindings = parsed_query.get("attribute_bindings", [])

        cand_categories = set((candidate.get("categories") or []))
        cand_colors = set((candidate.get("colors") or []))
        cand_styles = set((candidate.get("styles") or []))
        cand_envs = set((candidate.get("environments") or []))

        if strict:
            if query_categories and not query_categories.issubset(cand_categories):
                return False
            if query_colors and not query_colors.issubset(cand_colors):
                return False
            if query_styles and not (query_styles & cand_styles) and candidate.get("style_score", 0.0) <= 0.0:
                return False
            if query_environments and not (query_environments & cand_envs) and candidate.get("environment_score", 0.0) <= 0.0:
                return False

            # For strict compositional intent, enforce category + color co-presence.
            for binding in bindings:
                obj = binding.get("object")
                color = binding.get("color")
                if obj and obj not in cand_categories:
                    return False
                if color and color not in cand_colors:
                    return False
            return True

        # Relaxed mode: focus on practical category/color matching.
        # Style/environment are treated as soft hints only.
        if query_categories and not (query_categories & cand_categories):
            return False

        if query_categories and candidate.get("category_score", 0.0) < 0.3:
            return False

        if query_colors and not (query_colors & cand_colors):
            return False

        if bindings:
            binding_hits = 0
            for binding in bindings:
                obj = binding.get("object")
                color = binding.get("color")
                if obj in cand_categories and color in cand_colors:
                    binding_hits += 1
            # For relaxed mode, require at least one binding hit when bindings are present.
            return binding_hits > 0

        # If no explicit bindings are present, passing category/color checks is enough.
        return True

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

        self.last_retrieval_meta = {"mode": "default", "message": ""}

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
            meta = self.metadata_by_image_id.get(image_id)
            if meta is None:
                continue

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

        if not self._has_any_constraints(parsed_query):
            return bound_results

        strict_results = [
            item for item in bound_results
            if self._candidate_matches(item, parsed_query, strict=True)
        ]
        if strict_results:
            self.last_retrieval_meta = {
                "mode": "strict",
                "message": "Showing strict matches for all requested constraints.",
            }
            return strict_results

        relaxed_results = [
            item for item in bound_results
            if self._candidate_matches(item, parsed_query, strict=False)
        ]
        if relaxed_results:
            self.last_retrieval_meta = {
                "mode": "relaxed",
                "message": "No exact match found; showing closest relevant matches.",
            }
            return relaxed_results

        # Final conservative fallback: high-confidence category matches only.
        query_categories = set(parsed_query.get("categories", []))
        query_styles = set(parsed_query.get("styles", []))
        query_environments = set(parsed_query.get("environments", []))
        has_bindings = bool(parsed_query.get("attribute_bindings", []))
        if query_categories and not has_bindings:
            category_only = [
                item for item in bound_results
                if item.get("category_score", 0.0) >= 0.9
            ]
            if category_only:
                self.last_retrieval_meta = {
                    "mode": "relaxed",
                    "message": "No exact attribute match found; showing closest category matches.",
                }
                return category_only

        # Soft fallback for simple single-category queries (e.g., "white dress"):
        # if exact color/style/environment matching is sparse, still return the
        # closest high-confidence category hits.
        if len(query_categories) == 1 and not query_styles and not query_environments:
            simple_category_only = [
                item for item in bound_results
                if item.get("category_score", 0.0) >= 0.75
            ]
            if simple_category_only:
                self.last_retrieval_meta = {
                    "mode": "relaxed",
                    "message": "No exact match found; showing closest category results.",
                }
                return simple_category_only

        # Last resort fallback to keep UX usable: return ranked results instead
        # of empty list when strict constraints are too sparse/noisy.
        if bound_results:
            self.last_retrieval_meta = {
                "mode": "relaxed",
                "message": "Showing broad closest matches.",
            }
            return bound_results

        self.last_retrieval_meta = {
            "mode": "none",
            "message": "No strict or close match found for this query.",
        }
        return []