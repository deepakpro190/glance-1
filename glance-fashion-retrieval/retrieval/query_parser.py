import re

from indexing.parser import FashionpediaParser


class QueryParser:

    COLORS = {
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
        "gold",
        "silver",
        "navy",
        "beige",
        "tan",
        "olive",
        "maroon",
        "denim",
        "bright",
        "dark",
        "light",
    }

    STYLES = {
        "formal",
        "casual",
        "business",
        "party",
        "sport",
        "sports",
        "traditional",
        "ethnic",
        "wedding",
        "professional",
        "funeral",
        "office",
    }

    ENVIRONMENTS = {
        "office",
        "street",
        "park",
        "beach",
        "home",
        "city",
        "indoor",
        "outdoor",
        "modern",
        "formal",
        "casual",
    }

    SYNONYMS = {
        "blouse": "shirt",
        "shirt": "shirt",
        "shirts": "shirt",
        "dress": "dress",
        "dresses": "dress",
        "jacket": "jacket",
        "jackets": "jacket",
        "pants": "pants",
        "trouser": "pants",
        "trousers": "pants",
        "skirt": "skirt",
        "skirts": "skirt",
        "coat": "coat",
        "coats": "coat",
        "tie": "tie",
        "ties": "tie",
        "shoe": "shoe",
        "shoes": "shoe",
        "suit": "suit",
        "suits": "suit",
        "top": "top",
        "tops": "top",
        "sweater": "sweater",
        "sweaters": "sweater",
        "cardigan": "cardigan",
        "cardigans": "cardigan",
        "vest": "vest",
        "vests": "vest",
        "shorts": "shorts",
        "short": "shorts",
        "glasses": "glasses",
        "hat": "hat",
        "hats": "hat",
        "bag": "bag",
        "wallet": "wallet",
        "glove": "glove",
        "gloves": "glove",
        "watch": "watch",
        "belt": "belt",
        "leg warmer": "leg warmer",
        "tights": "tights",
        "stockings": "tights",
        "sock": "sock",
        "socks": "sock",
        "scarf": "scarf",
        "scarves": "scarf",
        "umbrella": "umbrella",
        "hood": "hood",
        "collar": "collar",
        "lapel": "lapel",
        "sleeve": "sleeve",
        "pocket": "pocket",
        "neckline": "neckline",
        "buckle": "buckle",
        "zipper": "zipper",
        "applique": "applique",
        "bead": "bead",
        "bow": "bow",
        "flower": "flower",
        "fringe": "fringe",
        "ribbon": "ribbon",
        "rivet": "rivet",
        "ruffle": "ruffle",
        "sequin": "sequin",
        "tassel": "tassel",
        "blouse": "shirt",
        "shirts": "shirt",
        "shirt": "shirt",
        "dresses": "dress",
        "dress": "dress",
        "ties": "tie",
        "tie": "tie",
        "skirts": "skirt",
        "skirt": "skirt",
        "coats": "coat",
        "coat": "coat",
        "jackets": "jacket",
        "jacket": "jacket",
        "pants": "pants",
        "trousers": "pants",
        "trouser": "pants",
        "shoes": "shoe",
        "shoe": "shoe",
        "suits": "suit",
        "suit": "suit",
        "formal": "formal",
        "business": "business",
        "casual": "casual",
        "office": "office",
        "city": "city",
        "park": "park",
        "beach": "beach",
        "street": "street",
        "funeral": "funeral",
    }

    ####################################################

    def __init__(self):
        self.parser = FashionpediaParser()
        self.categories = {c.lower() for c in self.parser.get_all_categories()}

    ####################################################

    def _normalize(self, word):
        word = word.strip("-").lower()
        if len(word) > 3 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us")):
            return word[:-1]
        return word

    ####################################################

    def parse(self, query):
        words = re.findall(r"[a-zA-Z\-]+", query.lower())
        normalized_words = []
        for word in words:
            if word in {"and", "a", "an", "the", "in", "inside", "for", "on", "with", "of"}:
                continue
            normalized_words.append(word)

        colors = []
        categories = []
        styles = []
        environments = []
        extracted_terms = set()

        for raw_word in normalized_words:
            word = self._normalize(raw_word)
            canonical = self.SYNONYMS.get(word, word)

            if word in self.COLORS or canonical in self.COLORS:
                color = canonical if canonical in self.COLORS else word
                colors.append(color)
                extracted_terms.add(color)

            if canonical in self.categories or word in self.categories:
                category = canonical if canonical in self.categories else word
                categories.append(category)
                extracted_terms.add(category)
            elif canonical in self.SYNONYMS and self.SYNONYMS[canonical] in self.categories:
                category = self.SYNONYMS[canonical]
                categories.append(category)
                extracted_terms.add(category)
            elif word in self.SYNONYMS and self.SYNONYMS[word] in self.categories:
                category = self.SYNONYMS[word]
                categories.append(category)
                extracted_terms.add(category)
            elif word in {"shirt", "dress", "jacket", "pants", "skirt", "coat", "tie", "shoe", "bag", "hat", "glasses", "sweater", "cardigan", "vest", "shorts", "watch", "belt", "scarf", "sock", "sleeve", "neckline", "collar"}:
                categories.append(word)
                extracted_terms.add(word)

            if word in self.STYLES or canonical in self.STYLES:
                style = canonical if canonical in self.STYLES else word
                styles.append(style)
                extracted_terms.add(style)

            if word in self.ENVIRONMENTS or canonical in self.ENVIRONMENTS:
                environment = canonical if canonical in self.ENVIRONMENTS else word
                environments.append(environment)
                extracted_terms.add(environment)

        return {
            "query": query,
            "colors": sorted(list(set(colors))),
            "categories": sorted(list(set(categories))),
            "styles": sorted(list(set(styles))),
            "environments": sorted(list(set(environments))),
            "extracted_terms": extracted_terms,
        }