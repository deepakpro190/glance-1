from collections import Counter


class BoundPairReranker:
    def __init__(self):
        self.color_aliases = {
            "navy": "blue",
            "denim": "blue",
            "tan": "brown",
            "olive": "green",
            "maroon": "red",
            "gold": "yellow",
        }

    def normalize_color(self, color):
        color = (color or "").lower()
        return self.color_aliases.get(color, color)

    def bound_pair_score(self, parsed_query, candidate):
        query_colors = parsed_query.get("colors", [])
        query_categories = parsed_query.get("categories", [])
        image_categories = [c.lower() for c in candidate.get("categories", [])]
        image_colors = [c.lower() for c in candidate.get("colors", [])]
        garment_pairs = candidate.get("garment_pairs", [])
        garment_attributes = candidate.get("garment_attributes", [])

        if not query_colors or not query_categories:
            return 0.0

        matched = 0
        total_pairs = 0

        for cat in query_categories:
            for color in query_colors:
                total_pairs += 1
                cat_norm = cat.lower()
                color_norm = self.normalize_color(color)
                candidate_matches = any(
                    cat_norm in pair[0].lower() or pair[0].lower() in cat_norm
                    for pair in garment_pairs
                ) and (
                    any(
                        color_norm in item.lower() or item.lower() in color_norm
                        for item in image_colors
                    ) or any(
                        color_norm in str(attr.get("category", "")).lower() or str(attr.get("category", "")).lower() in color_norm
                        for attr in garment_attributes
                    )
                )
                if candidate_matches:
                    matched += 1

        return matched / max(1, total_pairs)

    def rerank(self, parsed_query, candidates):
        results = []
        for candidate in candidates:
            bound = self.bound_pair_score(parsed_query, candidate)
            candidate["bound_pair_score"] = round(bound, 4)
            candidate["bound_pair_final_score"] = round(
                0.55 * candidate.get("embedding_score", 0.0)
                + 0.30 * bound
                + 0.15 * candidate.get("component_scores", {}).get("category", 0.0),
                4,
            )
            results.append(candidate)

        results.sort(key=lambda x: x["bound_pair_final_score"], reverse=True)
        return results
