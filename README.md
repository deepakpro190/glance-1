# Glance Fashion Retrieval (Final)

Multimodal fashion image retrieval on Fashionpedia using SigLIP + FAISS + structured reranking.

This README reflects the current code in this workspace.

## What Is Implemented

- SigLIP text/image embedding retrieval via FAISS.
- Ontology-driven query parsing with synonym normalization.
- Structured constraints: categories, colors, styles, environments, and simple color-object bindings.
- Multi-component reranking with score breakdown fields.
- Retrieval fallback modes exposed to UI/API:
    - `strict`: exact constraint satisfaction
    - `relaxed`: close constrained match
    - `none`: no reliable match found
- FastAPI backend and React (Vite) frontend demo.

## Current Pipeline

```mermaid
flowchart LR
        Q[Query text] --> P[QueryParser]
        P --> E[SigLIP text embedding]
        E --> F[FAISS top-k search]
        F --> C[Candidate metadata join]
        C --> R[FashionReranker + BoundPairReranker + LearnedReranker score blend]
        R --> M[Strict/Relaxed/None selector]
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

## Environment Setup

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

Build artifacts (skip this if artifacts are already present and valid):

```powershell
..\.venv\Scripts\python.exe build_embeddings.py
..\.venv\Scripts\python.exe build_metadata.py
..\.venv\Scripts\python.exe build_faiss.py
```

Run backend:

```powershell
..\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8001
```

In a second terminal, run frontend:

```powershell
cd ..\glance-fashion-retrieval\frontend
npm install
npm run dev
```

Open the local frontend URL shown by Vite and search. If the UI cannot connect, verify backend is live at http://127.0.0.1:8001/health.

From workspace root:

```powershell
Set-Location c:\glance\glance-fashion-retrieval
python -m venv c:\glance\.venv
c:\glance\.venv\Scripts\activate
```

Install backend dependencies:

```powershell
pip install torch transformers datasets faiss-cpu numpy pillow tqdm scikit-learn joblib fastapi "uvicorn[standard]" pytest
```

Frontend dependencies:

```powershell
Set-Location c:\glance\glance-fashion-retrieval\frontend
npm install
```

## Build / Rebuild Artifacts

From `c:\glance\glance-fashion-retrieval`:

```powershell
c:\glance\.venv\Scripts\python.exe build_embeddings.py
c:\glance\.venv\Scripts\python.exe build_metadata.py
c:\glance\.venv\Scripts\python.exe build_faiss.py
```

Notes:
- `build_metadata.py` uses image IDs from `artifacts/embeddings/image_ids.json`.
- Rebuild can take time because Fashionpedia is loaded through Hugging Face datasets.

## Run The System

Backend API:

```powershell
Set-Location c:\glance\glance-fashion-retrieval
c:\glance\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8001
```

Frontend:

```powershell
Set-Location c:\glance\glance-fashion-retrieval\frontend
npm run dev
```

CLI mode:

```powershell
Set-Location c:\glance\glance-fashion-retrieval
c:\glance\.venv\Scripts\python.exe main.py
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
Set-Location c:\glance\glance-fashion-retrieval
c:\glance\.venv\Scripts\python.exe -m pytest tests/test_retrieval_components.py -q
```

Run the simple evaluation script:

```powershell
Set-Location c:\glance\glance-fashion-retrieval
c:\glance\.venv\Scripts\python.exe evaluate.py
```

## Retrieval Behavior Notes

- Complex compositional constraints can intentionally return `none` when the index has no reliable match.
- `style_score` and `environment_score` can be low when metadata labels are sparse; the UI treats unrequested facets as `N/A`.
- Query quality depends heavily on the metadata artifact quality and category coverage in the indexed set.

## Troubleshooting

- `Failed to fetch` in frontend:
    - Verify backend is running on `127.0.0.1:8001`.
    - Check `VITE_BACKEND_URL` if you changed ports.
- Slow startup:
    - First run downloads/loads model and dataset caches.
- Empty results:
    - Check `retrieval_mode` and `message` from API to see if this is an intentional strict rejection.

## License

Educational/research usage. Dataset usage is subject to Fashionpedia licensing.
