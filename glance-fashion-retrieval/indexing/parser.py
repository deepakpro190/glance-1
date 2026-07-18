from dataclasses import dataclass
from typing import List

from datasets import load_dataset


# -----------------------------
# Data Classes
# -----------------------------
@dataclass
class DetectedObject:
    category_id: int
    category_name: str
    bbox: list
    area: float


@dataclass
class ImageMetadata:
    image_id: int
    image: object
    width: int
    height: int
    objects: List[DetectedObject]


# -----------------------------
# Parser
# -----------------------------

class FashionpediaParser:

    def __init__(self):

        print("Loading Fashionpedia dataset...")

        self.dataset = load_dataset(
            "detection-datasets/fashionpedia",
            cache_dir="./data"
        )

        self.train = self.dataset["train"]
        self.val = self.dataset["val"]
        self.category_names = (
        self.train.features["objects"]["category"]
        .feature
        .names
    )

        print(f"Loaded {len(self.category_names)} categories.")

        print("Dataset Loaded Successfully!")

    def get_sample(self, index):

        sample = self.train[index]

        detected_objects = []

        categories = sample["objects"]["category"]
        boxes = sample["objects"]["bbox"]
        areas = sample["objects"]["area"]

        for category_id, bbox, area in zip(
            categories,
            boxes,
            areas
        ):

            category_name = self.category_names[category_id]

            detected_objects.append(
                DetectedObject(
                    category_id=category_id,
                    category_name=category_name,
                    bbox=bbox,
                    area=area
                )
            )

        return ImageMetadata(
            image_id=sample["image_id"],
            image=sample["image"],
            width=sample["width"],
            height=sample["height"],
            objects=detected_objects
        )
    


    def get_all_categories(self):

        category_feature = self.train.features["objects"]["category"]

        return category_feature.feature.names