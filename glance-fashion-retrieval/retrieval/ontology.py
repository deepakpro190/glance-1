from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass(frozen=True)
class CategoryAlias:
    canonical: str
    aliases: List[str]


CATEGORY_ALIASES: List[CategoryAlias] = [
    CategoryAlias("shirt", ["shirt", "shirts", "tshirt", "t-shirt", "tee", "blouse"]),
    CategoryAlias("dress", ["dress", "dresses", "gown"]),
    CategoryAlias("pants", ["pants", "pant", "trouser", "trousers", "jeans", "jean", "denim"]),
    CategoryAlias("skirt", ["skirt", "skirts"]),
    CategoryAlias("jacket", ["jacket", "jackets", "blazer", "blazers"]),
    CategoryAlias("coat", ["coat", "coats", "raincoat", "overcoat"]),
    CategoryAlias("tie", ["tie", "ties", "necktie"]),
    CategoryAlias("shoe", ["shoe", "shoes", "sneaker", "sneakers", "heels", "heel", "boot", "boots"]),
    CategoryAlias("bag", ["bag", "bags", "handbag", "handbags", "purse"]),
    CategoryAlias("glasses", ["glasses", "spectacles", "eyewear"]),
    CategoryAlias("hat", ["hat", "hats", "cap", "caps"]),
    CategoryAlias("suit", ["suit", "suits", "business suit"]),
    CategoryAlias("sweater", ["sweater", "sweaters", "jumper"]),
    CategoryAlias("cardigan", ["cardigan", "cardigans"]),
    CategoryAlias("vest", ["vest", "vests", "waistcoat"]),
    CategoryAlias("shorts", ["short", "shorts"]),
    CategoryAlias("watch", ["watch", "watches"]),
    CategoryAlias("belt", ["belt", "belts"]),
    CategoryAlias("scarf", ["scarf", "scarves"]),
    CategoryAlias("sock", ["sock", "socks", "stockings", "tights"]),
]


COLOR_ALIASES: Dict[str, str] = {
    "navy": "blue",
    "denim": "blue",
    "maroon": "red",
    "burgundy": "red",
    "olive": "green",
    "tan": "brown",
    "beige": "brown",
    "gold": "yellow",
    "silver": "grey",
    "charcoal": "grey",
}


BASE_COLORS: Set[str] = {
    "black",
    "white",
    "red",
    "blue",
    "green",
    "yellow",
    "pink",
    "purple",
    "orange",
    "brown",
    "grey",
    "gray",
}


STYLE_LABELS: Set[str] = {
    "formal",
    "casual",
    "business",
    "party",
    "sports",
    "ethnic",
    "wedding",
    "professional",
}


STYLE_ALIASES: Dict[str, str] = {
    "funeral": "formal",
    "mourning": "formal",
    "office": "business",
    "corporate": "business",
    "work": "business",
    "weekend": "casual",
}


ENVIRONMENT_LABELS: Set[str] = {
    "office",
    "street",
    "park",
    "beach",
    "city",
    "indoor",
    "outdoor",
    "modern",
    "home",
}


def build_synonym_map() -> Dict[str, str]:
    synonym_map: Dict[str, str] = {}
    for alias in CATEGORY_ALIASES:
        for item in alias.aliases:
            synonym_map[item] = alias.canonical
    return synonym_map


def normalize_color_token(token: str) -> str:
    token = (token or "").strip().lower()
    if token == "gray":
        token = "grey"
    return COLOR_ALIASES.get(token, token)
