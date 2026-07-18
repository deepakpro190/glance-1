import re

from indexing.parser import FashionpediaParser
from retrieval.ontology import (
    BASE_COLORS,
    ENVIRONMENT_LABELS,
    STYLE_ALIASES,
    STYLE_LABELS,
    build_synonym_map,
    normalize_color_token,
)


class QueryParser:
    STOPWORDS = {
        "and", "a", "an", "the", "in", "inside", "for", "on", "with", "of", "to", "at", "by", "from", "wearing"
    }

    ####################################################

    def __init__(self):
        self.parser = FashionpediaParser()
        self.dataset_categories = {c.lower() for c in self.parser.get_all_categories()}
        self.synonyms = build_synonym_map()
        self.ontology_categories = set(self.synonyms.values())
        self.categories = self.ontology_categories | self.dataset_categories
        self.colors = set(BASE_COLORS)
        self.styles = set(STYLE_LABELS)
        self.environments = set(ENVIRONMENT_LABELS)

    ####################################################

    def _normalize(self, word):
        word = word.strip("-").lower()
        if len(word) > 3 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us")):
            return word[:-1]
        return word

    ####################################################

    def _tokenize(self, query):
        return re.findall(r"[a-zA-Z\-]+", query.lower())

    ####################################################

    def _canonicalize(self, token):
        normalized = self._normalize(token)
        return self.synonyms.get(normalized, normalized)

    ####################################################

    def _is_category_token(self, token):
        if token in self.ontology_categories:
            return True
        return any(token == c or token in c.split() for c in self.dataset_categories)

    ####################################################

    def _extract_bindings(self, tokens):
        bindings = []
        n = len(tokens)

        for i in range(n - 1):
            left = normalize_color_token(self._canonicalize(tokens[i]))
            right = normalize_color_token(self._canonicalize(tokens[i + 1]))

            if left in self.colors and self._is_category_token(right):
                bindings.append({"object": right, "color": left, "source": "adjacent"})

            if self._is_category_token(left) and right in self.colors:
                bindings.append({"object": left, "color": right, "source": "adjacent"})

        dedup = []
        seen = set()
        for item in bindings:
            key = (item["object"], item["color"])
            if key in seen:
                continue
            seen.add(key)
            dedup.append(item)

        return dedup

    ####################################################

    def parse(self, query):
        words = self._tokenize(query)
        filtered_words = [w for w in words if w not in self.STOPWORDS]

        colors = []
        categories = []
        styles = []
        environments = []
        extracted_terms = set()

        for raw in filtered_words:
            canonical = normalize_color_token(self._canonicalize(raw))
            style_token = STYLE_ALIASES.get(canonical, canonical)

            if canonical in self.colors:
                colors.append(canonical)
                extracted_terms.add(canonical)

            if self._is_category_token(canonical):
                categories.append(canonical)
                extracted_terms.add(canonical)

            if style_token in self.styles:
                styles.append(style_token)
                extracted_terms.add(style_token)

            if canonical in self.environments:
                environments.append(canonical)
                extracted_terms.add(canonical)

        bindings = self._extract_bindings(words)

        for pair in bindings:
            extracted_terms.add(pair["object"])
            extracted_terms.add(pair["color"])

        return {
            "query": query,
            "colors": sorted(list(set(colors))),
            "categories": sorted(list(set(categories))),
            "styles": sorted(list(set(styles))),
            "environments": sorted(list(set(environments))),
            "extracted_terms": extracted_terms,
            "object_entities": sorted(list(set(categories))),
            "attribute_bindings": bindings,
            "query_tokens": filtered_words,
        }
