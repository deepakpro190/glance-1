import json
import os

from tqdm import tqdm

from indexing.parser import FashionpediaParser


class MetadataBuilder:

    def __init__(self):
        self.parser = FashionpediaParser()

    def build(self, limit=None, image_ids=None):

        metadata = {}

        total = len(self.parser.train)

        if limit is not None:
            total = min(limit, total)

        if image_ids is not None:
            total = min(total, len(image_ids))

        print(f"\nBuilding metadata for {total} images...\n")

        for idx in tqdm(range(total), desc="Building Metadata"):

            sample = self.parser.get_sample(idx)
            if image_ids is not None:
                expected_image_id = int(image_ids[idx])
                if sample.image_id != expected_image_id:
                    # Keep metadata aligned with the same image ordering used by the embedding artifact.
                    sample = self.parser.get_sample(next(i for i, s in enumerate(self.parser.train) if s["image_id"] == expected_image_id))

            categories = []
            category_ids = []
            bboxes = []
            areas = []
            garment_types = []
            garment_colors = []
            garment_pairs = []
            garment_attributes = []
            garment_context = []
            style_hints = []
            colors = []
            styles = []
            environments = []

            for obj in sample.objects:
                category_name = obj.category_name.lower()
                categories.append(category_name)
                category_ids.append(obj.category_id)
                bboxes.append(obj.bbox)
                areas.append(obj.area)

                is_garment = any(
                    token in category_name
                    for token in ["shirt", "blouse", "dress", "tie", "coat", "jacket", "skirt", "pants", "shoe", "suit"]
                )

                if is_garment:
                    garment_types.append(category_name)
                    garment_pairs.append((category_name, "unknown"))
                    garment_attributes.append({
                        "category": category_name,
                        "area": float(obj.area),
                        "bbox": obj.bbox,
                    })
                    garment_context.append(category_name)

                    if any(token in category_name for token in ["dress", "shirt", "coat", "jacket", "suit", "skirt", "pants"]):
                        garment_context.append("fashion_item")

                if any(token in category_name for token in ["necklace", "belt", "bag", "wallet", "shoe", "sleeve", "collar", "lapel", "neckline"]):
                    garment_context.append(category_name)

                if any(token in category_name for token in ["formal", "business", "office", "party", "wedding", "funeral"]):
                    style_hints.append(category_name)

                color_tokens = ["red", "blue", "green", "yellow", "pink", "purple", "orange", "brown", "grey", "gray", "gold", "silver", "navy", "beige", "tan", "olive", "maroon", "denim", "bright", "dark", "light", "white", "black"]
                style_tokens = ["formal", "casual", "business", "party", "sport", "traditional", "ethnic", "wedding", "professional", "funeral", "office"]
                environment_tokens = ["office", "street", "park", "beach", "home", "city", "indoor", "outdoor", "modern"]

                matched_colors = [token for token in color_tokens if token in category_name]
                if matched_colors:
                    colors.extend(matched_colors)
                elif any(token in category_name for token in ["shirt", "blouse", "dress", "coat", "jacket", "skirt", "pants", "shoe"]):
                    colors.append("neutral")

                matched_styles = [token for token in style_tokens if token in category_name]
                if matched_styles:
                    styles.extend(matched_styles)
                elif any(token in category_name for token in ["shirt", "blouse", "dress", "coat", "jacket", "skirt", "pants", "shoe"]):
                    styles.append("casual")

                matched_environments = [token for token in environment_tokens if token in category_name]
                if matched_environments:
                    environments.extend(matched_environments)
                elif any(token in category_name for token in ["shirt", "blouse", "dress", "coat", "jacket", "skirt", "pants", "shoe"]):
                    environments.append("indoor")

            metadata[str(sample.image_id)] = {
                "image_id": sample.image_id,
                "width": sample.width,
                "height": sample.height,
                "categories": categories,
                "category_ids": category_ids,
                "bboxes": bboxes,
                "areas": areas,
                "garment_types": garment_types,
                "garment_colors": garment_colors,
                "garment_pairs": garment_pairs,
                "garment_attributes": garment_attributes,
                "garment_context": garment_context,
                "style_hints": style_hints,
                "colors": colors,
                "styles": styles,
                "environments": environments,
                "num_objects": len(categories),
            }

        return metadata

    def save(self, metadata, output_path):

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"\nMetadata saved successfully!")
        print(f"Location : {output_path}")
        print(f"Total Images : {len(metadata)}")