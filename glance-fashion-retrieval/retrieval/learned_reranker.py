import json
import os
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from config import ROOT_DIR


class LearnedReranker:
    def __init__(self, model_path=None):
        self.model_path = Path(model_path or ROOT_DIR / "artifacts" / "learned_reranker_model.joblib")
        self.vectorizer = DictVectorizer(sparse=False)
        self.model = None

    def _build_training_examples(self, metadata):
        examples = []
        labels = []

        for image_id, item in metadata.items():
            if not item.get("categories"):
                continue
            feature = {
                "num_objects": float(item.get("num_objects", 0)),
                "category_count": float(len(item.get("categories", []))),
                "garment_type_count": float(len(item.get("garment_types", []))),
                "has_dress": 1.0 if any("dress" in c.lower() for c in item.get("categories", [])) else 0.0,
                "has_shirt": 1.0 if any("shirt" in c.lower() for c in item.get("categories", [])) else 0.0,
                "has_tie": 1.0 if any("tie" in c.lower() for c in item.get("categories", [])) else 0.0,
                "has_shoe": 1.0 if any("shoe" in c.lower() for c in item.get("categories", [])) else 0.0,
                "has_color": 1.0 if item.get("colors") else 0.0,
                "has_style": 1.0 if item.get("styles") else 0.0,
                "has_environment": 1.0 if item.get("environments") else 0.0,
                "has_garment_types": 1.0 if item.get("garment_types") else 0.0,
            }
            examples.append(feature)
            labels.append(1 if item.get("num_objects", 0) >= 3 else 0)

        return examples, labels

    def fit(self, metadata):
        examples, labels = self._build_training_examples(metadata)
        if len(examples) < 2:
            raise ValueError("Not enough metadata to train a reranker")

        X = self.vectorizer.fit_transform(examples)
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X, labels)
        self._save_model()
        return self.model

    def _save_model(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump((self.vectorizer, self.model), self.model_path)

    def load(self):
        if not self.model_path.exists():
            return False
        import joblib
        self.vectorizer, self.model = joblib.load(self.model_path)
        return True

    def score(self, candidate, parsed_query):
        if self.model is None:
            return 0.0

        feature = {
            "num_objects": float(candidate.get("num_objects", 0)),
            "category_count": float(len(candidate.get("categories", []))),
            "garment_type_count": float(len(candidate.get("garment_types", []))),
            "has_dress": 1.0 if any("dress" in c.lower() for c in candidate.get("categories", [])) else 0.0,
            "has_shirt": 1.0 if any("shirt" in c.lower() for c in candidate.get("categories", [])) else 0.0,
            "has_tie": 1.0 if any("tie" in c.lower() for c in candidate.get("categories", [])) else 0.0,
            "has_shoe": 1.0 if any("shoe" in c.lower() for c in candidate.get("categories", [])) else 0.0,
            "has_color": 1.0 if candidate.get("colors") else 0.0,
            "has_style": 1.0 if candidate.get("styles") else 0.0,
            "has_environment": 1.0 if candidate.get("environments") else 0.0,
            "has_garment_types": 1.0 if candidate.get("garment_types") else 0.0,
            "query_has_color": 1.0 if parsed_query.get("colors") else 0.0,
            "query_has_style": 1.0 if parsed_query.get("styles") else 0.0,
            "query_has_environment": 1.0 if parsed_query.get("environments") else 0.0,
        }
        X = self.vectorizer.transform([feature])
        return float(self.model.predict_proba(X)[0][1])
