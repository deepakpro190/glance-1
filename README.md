# Glance Fashion Retrieval (Final)

Multimodal fashion image retrieval on Fashionpedia using SigLIP + FAISS + structured reranking.

This README reflects the current code in this workspace.

PICTURES OF WORKING OF PROJECT:

<img width="617" height="911" alt="Screenshot 2026-07-18 162000" src="https://github.com/user-attachments/assets/ed1f3fa5-68d6-482a-8252-09857b6ad1b8" />
<img width="1891" height="756" alt="Screenshot 2026-07-18 200041" src="https://github.com/user-attachments/assets/3f50cf5a-9879-4247-8fa2-8e1336f575f1" />
<img width="1773" height="916" alt="Screenshot 2026-07-18 200049" src="https://github.com/user-attachments/assets/496831e6-cde5-43b4-9d11-415199f23491" />
<img width="1851" height="935" alt="Screenshot 2026-07-18 200106" src="https://github.com/user-attachments/assets/359a3618-28b1-4760-b2c3-b60c7d1096db" />
<img width="1655" height="872" alt="Screenshot 2026-07-18 200114" src="https://github.com/user-attachments/assets/9cbac62e-c93c-4511-b665-a4a7dbcc9150" />




## Assignment Q&A (Max 5)

### 1) What baseline were we trying to beat?
Vanilla SigLIP/CLIP + FAISS nearest-neighbor retrieval using only text-image similarity. That baseline is strong for broad semantics, but weak on compositional constraints like color-object binding and query intent details.

### 2) Why did we use SigLIP + FAISS instead of training a new end-to-end model?
SigLIP gives strong zero-shot visual-language embeddings, and FAISS gives fast scalable nearest-neighbor retrieval. This matches the assignment goal of practical multimodal retrieval without expensive full-model retraining.

### 3) Why add query parsing and ontology normalization?
User prompts contain synonyms and structured intent (category, color, style, environment). Ontology-driven parsing converts free text into stable constraints so the system can reason beyond raw embedding similarity.

### 4) Why not rely only on strict filtering?
Strict-only retrieval improves precision but can return empty results too often due to sparse/noisy metadata. We use strict + relaxed fallback modes so users still get relevant results when exact matches are unavailable.

### 5) How is this better than vanilla retrieval in practice?
This system adds structured reranking and constraint-aware selection on top of embedding search. In short: better control over query intent, clearer score breakdowns, and more useful behavior for real-world queries than plain nearest-neighbor similarity alone.

## Assignment Compliance Snapshot

### Dataset and Coverage

- Source: Fashionpedia via Hugging Face datasets.
- Indexed scale: 45K+ images (well above the minimum 500-1000 requirement).
- Coverage dimensions present in metadata and retrieval logic:
    - Environment: office, street/city, park, indoor/home-like contexts.
    - Clothing types: formal, casual, outerwear, and related garment categories.
    - Color: normalized garment color terms used in query parsing and reranking.

### Vanilla Baseline vs Current Approach

| Aspect | Vanilla SigLIP/CLIP + FAISS | Current System |
|---|---|---|
| Query understanding | Raw text embedding only | Ontology + structured parser (color/category/style/environment/binding) |
| Compositionality | Weak (attribute mixing risk) | Binding-aware checks + reranking signals |
| Output behavior | Similarity-only nearest neighbors | Strict then relaxed retrieval with explicit mode/message |
| Explainability | Single similarity score | Component scores (category/color/style/env/object) |
| Practical UX | Can be brittle on sparse metadata | Returns closest useful results when strict match is unavailable |

### Required Evaluation Queries (Support Status)

| Query Type | Prompt | Current Handling |
|---|---|---|
| Attribute Specific | A person in a bright yellow raincoat. | Color + garment-category retrieval with relaxed fallback when exact pair is sparse |
| Contextual/Place | Professional business attire inside a modern office. | Style/environment terms parsed and used as soft constraints |
| Complex Semantic | Someone wearing a blue shirt sitting on a park bench. | Multi-attribute parsing with category/color/context-aware reranking |
| Style Inference | Casual weekend outfit for a city walk. | Style aliasing + environment hints + embedding relevance |
| Compositional | A red tie and a white shirt in a formal setting. | Strict path attempted first; falls back to relaxed when exact composition is unavailable |

### Zero-Shot, Scalability, and Modularity

- Modular workflows:
    - Part A Indexer: feature extraction + metadata + FAISS artifacts.
    - Part B Retriever: query parser + search + reranking + API/UI output.
- Zero-shot core: SigLIP embeddings and text-driven retrieval without end-to-end fine-tuning of the backbone.
- Current limitation note: a learned reranker module exists for pragmatic ranking support, so this is best described as mostly zero-shot retrieval with light supervised reranking.
- Scalability to 1M images (planned path):
    - Swap flat search to IVF/HNSW/PQ FAISS variants.
    - Keep metadata lookup O(1) using image-id keyed maps.
    - Use larger candidate pools with staged reranking.
    - Separate offline indexing jobs from online low-latency serving.

### Future Work Requested in Assignment

- Add richer location/weather understanding:
    - Extend taxonomy for city/place/weather labels.
    - Add location-weather prompt heads and calibrated fusion.
- Improve precision:
    - Better object-level color attribution and compositional grounding.
    - Query-adaptive weighting and stronger calibration.
    - Optional cross-encoder reranking for top-N precision.
- Improve evaluation quality:
    - Build labeled validation sets for per-query-type precision/recall.
    - Track strict-vs-relaxed retrieval metrics and user-facing success rates.

## What Is Implemented

- SigLIP text/image embedding retrieval via FAISS.
- Ontology-driven query parsing with synonym normalization.
- Structured constraints: categories, colors, styles, environments, and simple color-object bindings.
- Multi-component reranking with score breakdown fields.
- Retrieval fallback modes exposed to UI/API:
    - `strict`: exact constraint satisfaction
    - `relaxed`: closest useful match when strict fails
    - `none`: no reliable match available
- FastAPI backend and React (Vite) frontend demo.

## Current Pipeline

```mermaid
flowchart LR
        Q[Query text] --> P[QueryParser]
        P --> E[SigLIP text embedding]
        E --> F[FAISS top-k search]
        F --> C[Candidate metadata join]
        C --> R[FashionReranker + BoundPairReranker + LearnedReranker score blend]
    R --> M[Strict then Relaxed selector]
        M --> A[API response with component scores and retrieval message]
```

## Project Layout

```text
glance-fashion-retrieval/
    api.py
    main.py
    config.py
    build_embeddings.py
    build_metadata.py
    build_faiss.py
    evaluate.py
    retrieval/
        search_engine.py
        query_parser.py
        ontology.py
        reranker.py
        bound_pair_reranker.py
        learned_reranker.py
    indexing/
        parser.py
        embedder.py
        metadata_builder.py
        faiss_builder.py
    frontend/
        src/App.jsx
    artifacts/
        embeddings/
        metadata/
        faiss/
```

## Quick Start (Run On Any PC)

Clone and run with the same flow used in this repo:

```powershell
git clone https://github.com/deepakpro190/glance-1.git
cd glance-1\glance-fashion-retrieval
python -m venv ..\.venv
..\.venv\Scripts\activate
```

Install backend dependencies:

```powershell
pip install torch transformers datasets faiss-cpu numpy pillow tqdm scikit-learn joblib fastapi "uvicorn[standard]" pytest
```

Install frontend dependencies:

```powershell
cd frontend
npm install
cd ..
```

Build artifacts (skip this if artifacts are already present and valid):

```powershell
..\.venv\Scripts\python.exe build_embeddings.py
..\.venv\Scripts\python.exe build_metadata.py
..\.venv\Scripts\python.exe build_faiss.py
```

Run backend:

```powershell
..\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

In a second terminal, run frontend:

```powershell
cd frontend
npm run dev
```

Open the local frontend URL shown by Vite and search.

Backend health check:

```text
http://127.0.0.1:8000/health
```

Frontend currently tries `VITE_BACKEND_URL` first, and can fall back between localhost ports `8000` and `8001`.

## Optional CLI Mode

```powershell
..\.venv\Scripts\python.exe main.py
```

## API Contract

`GET /search?q=<query>` returns:

- `query`
- `parsed_terms`
- `parsed`
    - `colors`
    - `categories`
    - `styles`
    - `environments`
    - `attribute_bindings`
- `retrieval_mode` (`strict`, `relaxed`, `none`, or `default`)
- `message`
- `results[]`
    - `image_id`
    - `final_score`
    - `embedding_score`
    - `category_score`
    - `color_score`
    - `style_score`
    - `environment_score`
    - `object_score`
    - `categories`
    - `component_scores`
    - `image_url`

## Quick Quality Checks

Run tests:

```powershell
..\.venv\Scripts\python.exe -m pytest tests/test_retrieval_components.py -q
```

Run the simple evaluation script:

```powershell
..\.venv\Scripts\python.exe evaluate.py
```

## Retrieval Behavior Notes

- Color and category queries are intentionally relaxed to avoid empty outputs.
- Complex compositional queries may still return `none` if no reliable match exists.
- `style_score` and `environment_score` can be low when metadata labels are sparse; the UI treats unrequested facets as `N/A`.
- Query quality depends heavily on the metadata artifact quality and category coverage in the indexed set.

## Troubleshooting

- `Failed to fetch` in frontend:
    - Verify backend is running on `127.0.0.1:8000`.
    - Check `VITE_BACKEND_URL` if you changed ports.
    - Restart frontend dev server after changing backend URL settings.
- Slow startup:
    - First run downloads/loads model and dataset caches.
- Empty results:
    - Check `retrieval_mode` and `message` from API for strict vs relaxed fallback status.

## License

Educational/research usage. Dataset usage is subject to Fashionpedia licensing.
