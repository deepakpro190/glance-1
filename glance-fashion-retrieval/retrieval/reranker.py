from config import (
    EMBEDDING_WEIGHT,
    CATEGORY_WEIGHT,
    COLOR_WEIGHT,
    STYLE_WEIGHT,
    ENVIRONMENT_WEIGHT,
    OBJECT_WEIGHT,
)


class FashionReranker:

    def __init__(self):
        self.embedding_weight = EMBEDDING_WEIGHT
        self.category_weight = CATEGORY_WEIGHT
        self.color_weight = COLOR_WEIGHT
        self.style_weight = STYLE_WEIGHT
        self.environment_weight = ENVIRONMENT_WEIGHT
        self.object_weight = OBJECT_WEIGHT

    ###############################################################

    def category_score(self, query_categories, image_categories):
        if len(query_categories) == 0:
            return 0.0

        normalized_query = {
            c.lower().replace("-", " ").strip()
            for c in query_categories
            if c and str(c).strip()
        }
        normalized_image = {
            c.lower().replace("-", " ").replace(",", " ").strip()
            for c in image_categories
            if c and str(c).strip()
        }

        if not normalized_query:
            return 0.0

        exact_matches = 0
        partial_matches = 0

        for cat in normalized_query:
            if cat in normalized_image:
                exact_matches += 1
                continue
            for candidate in normalized_image:
                if cat in candidate or candidate in cat:
                    partial_matches += 1
                    break

        if exact_matches > 0:
            return min(1.0, (exact_matches + 0.5 * partial_matches) / len(normalized_query))
        return partial_matches / max(1, len(normalized_query))

    ###############################################################

    def color_score(self, query_colors, image_colors):
        if len(query_colors) == 0:
            return 0.0

        image_colors = {c.lower() for c in image_colors}
        matched = 0

        for color in query_colors:
            if color.lower() in image_colors:
                matched += 1

        return matched / len(query_colors)

    ###############################################################

    def style_score(self, query_styles, image_styles):
        if len(query_styles) == 0:
            return 0.0

        image_styles = {c.lower() for c in image_styles}
        matched = 0

        for style in query_styles:
            if style.lower() in image_styles:
                matched += 1

        if matched > 0:
            return matched / len(query_styles)

        # Fallback inference when explicit style tags are sparse in metadata.
        return 0.0

    ###############################################################

    def environment_score(self, query_environments, image_environments):
        if len(query_environments) == 0:
            return 0.0

        image_environments = {c.lower() for c in image_environments}
        matched = 0

        for environment in query_environments:
            if environment.lower() in image_environments:
                matched += 1

        return matched / len(query_environments)

    ###############################################################

    def object_score(self, num_objects):
        """
        Small bonus for richer fashion scenes.
        Caps at 1.0 after 10 objects.
        """
        return min(num_objects / 10.0, 1.0)

    ###############################################################

    def rerank(self, parsed_query, candidates):
        results = []

        for index, candidate in enumerate(candidates):
            candidate.setdefault("image_id", index)

            embedding = candidate["embedding_score"]
            category = self.category_score(
                parsed_query.get("categories", []),
                candidate.get("categories", [])
            )
            color = self.color_score(
                parsed_query.get("colors", []),
                candidate.get("colors", [])
            )
            style = self.style_score(
                parsed_query.get("styles", []),
                candidate.get("styles", [])
            )
            if style == 0.0 and parsed_query.get("styles", []):
                style = self._infer_style_from_categories(
                    parsed_query.get("styles", []),
                    candidate.get("categories", []),
                )
            environment = self.environment_score(
                parsed_query.get("environments", []),
                candidate.get("environments", [])
            )
            objects = self.object_score(candidate.get("num_objects", 0))

            overlap_bonus = 0.0
            if category >= 0.8:
                overlap_bonus += 0.15
            if color > 0.0:
                overlap_bonus += 0.12
            if style > 0.0:
                overlap_bonus += 0.10
            if environment > 0.0:
                overlap_bonus += 0.10
            if category == 1.0 and embedding > 0.04:
                overlap_bonus += 0.08
            if category == 1.0 and objects >= 0.5:
                overlap_bonus += 0.05

            final_score = (
                self.embedding_weight * embedding
                + self.category_weight * category
                + self.color_weight * color
                + self.style_weight * style
                + self.environment_weight * environment
                + self.object_weight * objects
                + overlap_bonus
            )

            candidate["category_score"] = round(category, 4)
            candidate["color_score"] = round(color, 4)
            candidate["style_score"] = round(style, 4)
            candidate["environment_score"] = round(environment, 4)
            candidate["object_score"] = round(objects, 4)
            candidate["raw_final_score"] = round(final_score, 6)
            candidate["raw_embedding_score"] = round(embedding, 6)
            candidate["component_scores"] = {
                "embedding": round(embedding, 4),
                "category": round(category, 4),
                "color": round(color, 4),
                "style": round(style, 4),
                "environment": round(environment, 4),
                "object": round(objects, 4),
            }

            results.append(candidate)

        results.sort(key=lambda x: x["raw_final_score"], reverse=True)

        best_embedding = max((item["raw_embedding_score"] for item in results), default=0.0)
        best_final = max((item["raw_final_score"] for item in results), default=0.0)

        for item in results:
            embedding_ratio = item["raw_embedding_score"] / max(best_embedding, 1e-8)
            final_ratio = item["raw_final_score"] / max(best_final, 1e-8)
            item["embedding_score"] = round(90.0 + 10.0 * min(1.0, embedding_ratio), 2)
            item["final_score"] = round(90.0 + 10.0 * min(1.0, final_ratio), 2)

        return results

    ###############################################################

    def _infer_style_from_categories(self, query_styles, image_categories):
        categories = {str(c).lower().strip() for c in image_categories if c}
        if not categories:
            return 0.0

        style_category_map = {
            "formal": {"dress", "shirt", "tie", "blazer", "jacket", "trousers", "suit"},
            "casual": {"t shirt", "jeans", "shorts", "sneakers", "hoodie"},
            "sporty": {"sneakers", "leggings", "tracksuit", "jersey"},
            "party": {"dress", "heels", "skirt", "blouse"},
        }

        best = 0.0
        for style in query_styles:
            hints = style_category_map.get(str(style).lower(), set())
            if not hints:
                continue
            if any((h in cat) or (cat in h) for h in hints for cat in categories):
                best = max(best, 0.35)

        return best