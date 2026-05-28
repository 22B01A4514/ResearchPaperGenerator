# AI Research Paper Generator

An end-to-end system that automatically generates IEEE-format research papers using Retrieval-Augmented Generation (RAG), a GNN-style Knowledge Graph, semantic citation injection, and real dataset-driven visualizations.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Module Reference](#module-reference)
- [Evaluation Metrics](#evaluation-metrics)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [License](#license)

---

## Overview

This system takes a research topic (and optionally a project description + uploaded reference papers) and produces a fully structured IEEE two-column `.docx` paper with:

- Eight sections (Abstract → Conclusion)
- Inline citations injected via pure semantic similarity (cosine distance, no keyword matching)
- Data-driven graphs generated from real Kaggle / UCI datasets
- NLP quality metrics: Perplexity, BLEU, ROUGE-1/2/L
- A GNN-style knowledge graph built from the retrieved papers

The backend is a **FastAPI** server; the frontend is a **React (Vite)** single-page app.

---

## Architecture

```
User (React SPA)
      │
      ▼
FastAPI (main.py)
      │
      ├─► AdvancedRAGEngine        (rag_engine.py)
      │     └─ FAISS IVF-PQ index over 1.7 M arXiv papers
      │         all-mpnet-base-v2 embeddings (768-dim)
      │
      ├─► LLMPaperGenerator        (llm_generator.py)
      │     ├─ KnowledgeGraph (GNN-style: paper/method/dataset nodes)
      │     └─ Groq LLaMA-3.1-8b-instant section generation
      │
      ├─► AdvancedCitationManager  (citation_manager.py)
      │     └─ Pure semantic citation injection (cosine ≥ 0.38)
      │
      ├─► DatasetDiscoveryEngine   (dataset_discovery.py)
      │     ├─ Live UCI ML Repository API
      │     └─ Kaggle search API
      │
      ├─► GraphGenerator           (graph_generator.py)
      │     └─ Matplotlib / Seaborn visualizations
      │         (heatmap, scatter matrix, distributions, …)
      │
      ├─► evaluate_paper()         (citation_manager.py)
      │     └─ Perplexity, BLEU, ROUGE-1/2/L
      │
      └─► build_ieee_docx()        (docx_builder.py)
            └─ Two-column IEEE .docx with figures + metrics appendix
```

---

## Features

### Core Pipeline

| Feature | Detail |
|---|---|
| RAG Retrieval | FAISS IVF-PQ, up to 1.7 M arXiv papers, `all-mpnet-base-v2` (768-dim) |
| Knowledge Graph | GNN-style: paper, method, dataset nodes; `similar_to` edges via cosine similarity |
| Section generation | 8 IEEE sections via Groq LLaMA-3.1-8b-instant |
| Citation injection | Pure embedding cosine similarity — no regex, no keyword lists |
| Dataset discovery | Live UCI / OpenML API + Kaggle search API |
| Graph generation | 6 chart types from real DataFrames (heatmap, scatter, distribution, box, class dist., feature importance) |
| DOCX output | IEEE two-column layout, dynamic index terms, algorithm pseudocode blocks, system architecture table |
| NLP metrics | Bigram-perplexity, BLEU-4, ROUGE-1/2/L, lexical diversity |
| Multi-approach | Uploaded / RAG / Hybrid generation comparison with precision/recall/F1 |
| vs-baseline | Side-by-side metric comparison against a plain GPT-style baseline |

### Frontend Features

- Onboarding wizard (fast mode vs. full analysis)
- Project context form (tech stack, algorithms, contributions, etc.)
- Dataset relevance gate with per-graph analysis
- Hallucination highlighter (grounded vs. unsupported claims)
- Per-section Flesch-Kincaid readability scores
- BibTeX export
- Citation strength checker
- Research gap detector (SBERT sparse-paper detection)

---

## Tech Stack

**Backend**

- Python 3.10+
- FastAPI + Uvicorn
- `sentence-transformers` (`all-mpnet-base-v2`, `all-MiniLM-L6-v2`)
- FAISS (`faiss-cpu` or `faiss-gpu`)
- `rank-bm25` (Okapi BM25 fallback)
- Groq Python SDK (`llama-3.1-8b-instant`)
- `python-docx` (IEEE DOCX builder)
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- `pdfplumber`, `PyPDF2` (paper extraction)
- `scipy` (p-values, Shapiro-Wilk normality test)
- `citeproc-py` (CSL bibliography, optional)
- `nltk`, `wordnet` (dataset discovery NLP)

**Frontend**

- React 18 + Vite
- Chart.js 4 (bar, radar charts)
- Vanilla CSS custom design tokens (no UI framework)

---

## Installation

### 1. Clone

```bash
git clone https://github.com/22B01A4514/ResearchPaperGenerator.git
cd ResearchPaperGenerator
```

### 2. Python environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If `requirements.txt` is not yet present, install manually:

```bash
pip install fastapi uvicorn python-multipart python-dotenv \
            groq sentence-transformers faiss-cpu rank-bm25 \
            pandas numpy matplotlib seaborn scikit-learn scipy \
            python-docx pdfplumber PyPDF2 nltk requests tqdm \
            citeproc-py kaggle
```

### 3. Frontend

```bash
cd frontend          # adjust to your actual frontend directory
npm install
npm run dev          # starts at http://localhost:5173
```

### 4. Download the arXiv dataset

```bash
python kaggle_downloader.py
```

This downloads `arxiv-metadata-oai-snapshot.json` (~3.5 GB) from the Cornell-University/arxiv Kaggle dataset into `data/`.

---

## Configuration

Create a `.env` file in the project root:

```env
# Groq API key — https://console.groq.com
API_KEY=gsk_...

# Kaggle credentials — https://www.kaggle.com/settings (API token)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

---

## Usage

### Start the backend

```bash
uvicorn main:app --reload --port 8000
```

The first startup builds the FAISS index (10,000 papers by default; edit `max_papers` in `main.py → get_rag()` for more). Index and embeddings are cached under `data/cache/`.

### Start the frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

### Generate a paper (minimal)

1. Enter a research topic.
2. Choose a reference source (arXiv RAG / Upload / Both).
3. Optionally fill in the Project Context form.
4. Click **Generate Research Paper**.
5. Download the `.docx` from the Generated Paper panel.

### Generate a paper with graphs

1. Check **Include data-driven graphs**.
2. Select a dataset from the discovered list.
3. Pick graph types and click **Generate Graphs**.
4. Then click **Generate Research Paper**.

---

## Module Reference

### `rag_engine.py` — AdvancedRAGEngine

Loads arXiv papers, encodes them with `all-mpnet-base-v2`, and stores them in a FAISS IVF-PQ index.

```python
engine = AdvancedRAGEngine()
engine.load_dataset(max_papers=10000)
engine.build_index()
papers = engine.search("transformer attention mechanism", top_k=10)
```

Index type selection:
- `n < 256` → `IndexFlatIP` (exact)
- `n ≥ 256` → `IndexIVFPQ` with `nlist = min(4096, 4·√n)` and `m = 32` sub-quantizers

### `llm_generator.py` — LLMPaperGenerator + KnowledgeGraph

```python
llm = LLMPaperGenerator()
kg  = llm.build_knowledge_graph(papers)       # builds GNN-style graph
text = llm.generate_section(
    section_type="methodology",
    topic="Skin cancer detection using CNNs",
    retrieved_papers=papers,
    project_context={"title": "...", "summary": "...", "features": {...}},
)
```

The `KnowledgeGraph` extracts method/dataset nodes from paper text using regex patterns and builds `similar_to` edges with cosine similarity ≥ 0.5.

### `citation_manager.py` — AdvancedCitationManager

```python
mgr = AdvancedCitationManager(CitationFormat.IEEE)
for paper in papers:
    mgr.add_paper(paper)

enriched_text, report = mgr.insert_citations_smart(section_text, section_type="introduction")
bibliography = mgr.format_bibliography()
metrics = evaluate_paper(full_text, reference_abstracts)
```

Citation injection thresholds:

| Section | Density target |
|---|---|
| abstract | 0 (never cite) |
| introduction | 0.30 |
| literature_survey | 0.55 |
| methodology | 0.35 |
| results | 0.30 |
| conclusion | 0.20 |

Semantic similarity threshold: **0.38** (injection) / **0.42** (grounded claim).

### `dataset_discovery.py` — DatasetDiscoveryEngine

```python
engine = DatasetDiscoveryEngine(kaggle_username="...", kaggle_key="...")
datasets = engine.discover_datasets("skin cancer classification", top_k=8)
df, meta = engine.load_dataset(datasets[0])
```

Discovery strategy: Kaggle live search → UCI/OpenML catalog (hybrid BM25 + SBERT) → fallback padding.

All field weights, stopwords, column names, and domain suggestions are derived at runtime — no static hardcoded lists.

### `graph_generator.py` — GraphGenerator

```python
gen = GraphGenerator()
graphs = gen.generate(
    df=df, metadata=meta,
    selected_types=["correlation_heatmap", "feature_importance"],
    dataset_title="ISIC Skin Lesion Dataset",
    topic="skin cancer detection",
)
# Each graph dict has: id, title, data (base64 PNG), statistical_insight, stats
```

Also exposes `quality_report(df)` returning completeness %, imbalance ratio, collinearity score, outlier density, and an A/B/C grade.

### `docx_builder.py` — build_ieee_docx

```python
build_ieee_docx(
    output_path="paper.docx",
    topic="Skin Cancer Detection using Deep Learning",
    sections={"abstract": "...", "introduction": "...", ...},
    citations=[{"id": 1, "text": "[1] Author, Title, Journal, Year."}],
    graphs=[{"id": "fig_1", "data": "", "title": "Heatmap", ...}],
    dataset_title="ISIC 2019",
    metrics={"perplexity": 42.1, "bleu_score": 0.38, ...},
)
```

Index terms are derived dynamically from topic + abstract + introduction via TF-IDF frequency analysis — never hardcoded.

### `paper_extractor.py` — extract_papers

```python
papers = extract_papers(["paper1.pdf", "paper2.docx"], deduplicate=True)
# Each dict: id, title, authors, year, abstract, full_text, chunks, extraction_confidence
```

Chunking: 400-token overlapping chunks (50-token overlap) stored in `paper["chunks"]` for fine-grained RAG retrieval.

---

## Evaluation Metrics

`evaluate_paper(generated_text, reference_texts)` returns:

| Metric | Description | Good threshold |
|---|---|---|
| Perplexity | Bigram LM log-probability | < 50 |
| BLEU-4 | n-gram precision vs references | ≥ 0.30 |
| ROUGE-1 F1 | Unigram overlap | ≥ 0.40 |
| ROUGE-2 F1 | Bigram overlap | ≥ 0.25 |
| ROUGE-L F1 | Longest common subsequence | ≥ 0.30 |
| Lexical diversity | Type-token ratio | ≥ 0.50 |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/datasets/discover` | Discover datasets for a topic |
| POST | `/datasets/load` | Load a dataset into session |
| GET | `/datasets/quality` | Dataset quality report (A/B/C grade) |
| POST | `/graphs/generate` | Generate graphs from loaded dataset |
| POST | `/graphs/analyze-relevance` | AI relevance analysis for one graph |
| POST | `/graphs/analyze-all-relevance` | Batch relevance analysis |
| POST | `/upload-papers` | Extract metadata from uploaded PDFs/DOCXs |
| POST | `/generate-paper` | Full paper generation (main endpoint) |
| GET | `/download/{filename}` | Download generated DOCX |
| POST | `/evaluate/citation` | Evaluate citation strength for a claim |
| POST | `/evaluate/section` | Score a section on 5 quality criteria |
| POST | `/evaluate/metrics` | Compute Perplexity / BLEU / ROUGE |
| POST | `/research-gaps` | Detect research gaps from paper corpus |
| POST | `/compare/multi-approach` | Generate Uploaded / RAG / Hybrid comparison |
| POST | `/compare/system-vs-baseline` | Metric comparison vs GPT-style baseline |
| GET | `/compare/tools` | Comparative analysis vs other tools |
| POST | `/session/clear-cache` | Clear LLM cache and session |
| GET | `/health` | Health check and feature list |

---

## Project Structure

```
ResearchPaperGenerator/
├── main.py                  # FastAPI app, all endpoints
├── rag_engine.py            # FAISS RAG over arXiv
├── llm_generator.py         # Groq LLM section generation + KnowledgeGraph
├── citation_manager.py      # Semantic citation injection + NLP metrics
├── dataset_discovery.py     # Live UCI/Kaggle dataset discovery
├── graph_generator.py       # Matplotlib/Seaborn chart generation
├── docx_builder.py          # IEEE two-column DOCX builder
├── paper_extractor.py       # PDF/DOCX metadata extraction + chunking
├── kaggle_downloader.py     # One-time arXiv dataset download script
├── .env                     # API keys (not committed)
├── data/
│   ├── arxiv-metadata-oai-snapshot.json   # ~3.5 GB arXiv dataset
│   └── cache/
│       ├── papers.pkl
│       ├── embeddings.npy
│       └── faiss_index.bin
└── frontend/
    ├── src/
    │   └── App.jsx          # React SPA (all components in one file)
    ├── package.json
    └── vite.config.js
```

---

## Limitations

- **FAISS index build time**: Encoding 1.7 M papers takes approximately 2–3 hours on CPU. Start with `max_papers=10000` for testing.
- **Groq rate limits**: The free tier has token-per-minute limits. Long papers may hit rate limits; retry logic (3 attempts with exponential back-off) is built in.
- **Graph generation**: Requires a loadable dataset. Some UCI/Kaggle datasets may be unavailable due to access restrictions or format changes.
- **Citeproc**: Full CSL bibliography formatting requires `citeproc-py`. Falls back to a clean IEEE format string if unavailable.
- **LLM accuracy**: Section content is grounded in retrieved papers, but the model can still hallucinate details. Use the Hallucination Highlighter panel and Evidence Alignment scores to review claims.

---

## License

This project is released for academic and educational use. See [LICENSE](LICENSE) for details.
