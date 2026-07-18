from pathlib import Path
import torch

# ==========================================================
# Project Paths
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent

ARTIFACTS_DIR = ROOT_DIR / "artifacts"

METADATA_DIR = ARTIFACTS_DIR / "metadata"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
FAISS_DIR = ARTIFACTS_DIR / "faiss"

METADATA_PATH = METADATA_DIR / "image_metadata.json"

EMBEDDINGS_PATH = EMBEDDINGS_DIR / "image_embeddings.npy"

IMAGE_IDS_PATH = EMBEDDINGS_DIR / "image_ids.json"

FAISS_INDEX_PATH = FAISS_DIR / "image_index.faiss"

# ==========================================================
# Model
# ==========================================================

MODEL_NAME = "google/siglip-base-patch16-224"

# ==========================================================
# Device
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# Embedding
# ==========================================================

BATCH_SIZE = 64

# ==========================================================
# Retrieval
# ==========================================================

TOP_K = 100

FINAL_RESULTS = 10

# ==========================================================
# Reranking Weights
# ==========================================================

EMBEDDING_WEIGHT = 0.60

CATEGORY_WEIGHT = 0.20

COLOR_WEIGHT = 0.10

STYLE_WEIGHT = 0.05

ENVIRONMENT_WEIGHT = 0.03

OBJECT_WEIGHT = 0.02