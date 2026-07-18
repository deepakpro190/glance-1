import json
from pathlib import Path

from retrieval.query_parser import QueryParser
from retrieval.reranker import FashionReranker


def test_metadata_artifact_matches_embedding_ids_artifact():
    root = Path(__file__).resolve().parents[1]
    image_ids_path = root / "artifacts" / "embeddings" / "image_ids.json"
    metadata_path = root / "artifacts" / "metadata" / "image_metadata.json"

    assert image_ids_path.exists(), "embedding ID artifact is missing"
    assert metadata_path.exists(), "metadata artifact is missing"

    image_ids = json.loads(image_ids_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    metadata_ids = {int(item["image_id"]) for item in metadata.values()}

    assert len(image_ids) == len(metadata_ids)
    assert set(image_ids).issubset(metadata_ids)


def test_query_parser_exposes_parser_for_image_lookup():
    parser = QueryParser()

    assert hasattr(parser, "parser")

    sample = parser.parser.get_sample(0)
    image = getattr(sample, "image", None)

    assert image is not None


def test_query_parser_extracts_shirt_category_from_synonym():
    parser = QueryParser()
    parsed = parser.parse("blue shirt sitting on a park bench")

    assert "park" in parsed["environments"]
    assert parsed["extracted_terms"] >= {"blue", "park", "shirt"}


def test_query_parser_extracts_structured_terms():
    parser = QueryParser()
    parsed = parser.parse("A red tie and a white shirt in a formal office")

    assert "red" in parsed["colors"]
    assert "white" in parsed["colors"]
    assert "tie" in parsed["categories"]
    assert "shirt" in parsed["categories"]
    assert "formal" in parsed["styles"]
    assert "office" in parsed["environments"]
    assert parsed["extracted_terms"] >= {"red", "tie", "shirt", "formal", "office"}


def test_reranker_uses_component_scores_for_final_rank():
    reranker = FashionReranker()
    parsed_query = {
        "colors": ["red"],
        "categories": ["tie"],
        "styles": ["formal"],
        "environments": ["office"],
    }

    candidates = [
        {
            "embedding_score": 0.70,
            "categories": ["tie"],
            "colors": ["red"],
            "styles": ["formal"],
            "environments": ["office"],
            "num_objects": 6,
        },
        {
            "embedding_score": 0.95,
            "categories": ["dress"],
            "colors": ["blue"],
            "styles": ["casual"],
            "environments": ["street"],
            "num_objects": 2,
        },
    ]

    results = reranker.rerank(parsed_query, candidates)

    assert results[0]["image_id"] == 0
    assert results[0]["component_scores"]["category"] == 1.0
    assert results[0]["component_scores"]["style"] == 1.0
    assert results[0]["component_scores"]["environment"] == 1.0
    assert results[0]["final_score"] > results[1]["final_score"]
