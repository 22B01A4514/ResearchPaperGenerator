"""
main.py — Explainable AI Research Paper Generator (v6)
=======================================================
FIXES in v6:

1. 422 ERROR on /graphs/analyze-relevance:
   - Root cause: GraphAnalysisRequest model required `graph_type` and
     `graph_description` fields, but the frontend was not always sending them.
   - Fix: Both fields now have default values (Optional with defaults).
   - Also added proper validation error logging.

2. GNN PIPELINE INTEGRATION:
   - LLMPaperGenerator.build_knowledge_graph() called ONCE before section generation.
   - KG context injected into every section prompt.
   - Methodology section specifically explains the pipeline stages.

3. METRICS — Perplexity, BLEU, ROUGE:
   - After paper generation, evaluate_paper() called on the full generated text.
   - Metrics stored in response and embedded in DOCX appendix.
   - /evaluate/metrics endpoint added.

4. SEMANTIC CITATION — no keyword matching:
   - AdvancedCitationManager v4 uses pure cosine similarity.
   - No regex triggers. Works for synonyms, paraphrases, domain shifts.

5. DOCX FIXES:
   - Abstract now properly single-column.
   - Graphs sized at 3.2" (fits two-column layout).
   - Metrics appendix added.
"""

import os
import json
import traceback
import uuid
import tempfile
import shutil
from dataclasses import fields as dataclass_fields
from typing import List, Optional, Dict

from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ── Local modules ───────────────────────────────────────────────────────────────
from llm_generator import (
    LLMPaperGenerator,
    KnowledgeGraph,
    evaluate_citation,
    generate_multi_approach_comparison,
    analyze_graph_relevance,
    generate_system_comparison_text,
    compare_system_vs_baseline,
    evaluate_section_quality,
    generate_section_with_analysis,
    detect_research_gaps,
    link_graphs_to_text,
    clear_cache as clear_llm_cache,
)
from rag_engine import AdvancedRAGEngine
from citation_manager import AdvancedCitationManager, CitationFormat, evaluate_paper
from dataset_discovery import (
    DatasetDiscoveryEngine,
    DatasetInfo,
    _get_live_catalog,
)
from graph_generator import GraphGenerator, quality_report
from docx_builder import build_ieee_docx
from paper_extractor import extract_papers, clear_duplicate_cache

# ── FastAPI ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="Explainable AI Research Paper Generator", version="6.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ──────────────────────────────────────────────────────────────────
dataset_engine = DatasetDiscoveryEngine(
    kaggle_username=os.getenv("KAGGLE_USERNAME"),
    kaggle_key=os.getenv("KAGGLE_KEY"),
)
graph_gen   = GraphGenerator()
_rag: Optional[AdvancedRAGEngine] = None
_sessions: Dict[str, Dict] = {}

GRAPH_SAMPLE_LIMIT = 5000


def get_rag() -> AdvancedRAGEngine:
    global _rag
    if _rag is None:
        _rag = AdvancedRAGEngine()
        _rag.load_dataset(max_papers=10000)
        _rag.build_index()
    return _rag


def _make_empty_session() -> Dict:
    return {
        "df": None, "metadata": None, "dataset_info": None,
        "graphs": [], "graph_context": "", "uploaded_papers": [],
        "topic": "", "project_context": None,
    }


def _datasetinfo_from_catalog(entry: Dict) -> DatasetInfo:
    valid_keys = {f.name for f in dataclass_fields(DatasetInfo)}
    clean = {k: v for k, v in entry.items() if k in valid_keys}
    clean.setdefault("relevance_score", 1.0)
    clean.setdefault("match_explanation", "exact catalog match")
    return DatasetInfo(**clean)


# ── Pydantic models ─────────────────────────────────────────────────────────────

class DatasetCard(BaseModel):
    id: str; source: str; title: str; description: str
    rows: Optional[int] = None; columns: Optional[int] = None
    size_mb: Optional[float] = None
    tags: List[str]; relevance_score: float

class DatasetLoadResponse(BaseModel):
    session_id: str; rows: int; original_rows: int; sampled: bool
    columns: int; numeric_cols: List[str]; categorical_cols: List[str]
    target_col: Optional[str] = None; missing_pct: float
    recommended_graphs: List[Dict]; preview: Dict

class GraphResult(BaseModel):
    id: str; type: str; title: str; figure_label: str; data: str
    statistical_insight: str; project_insight: str; insight: str; stats: Dict

class UploadedPaperInfo(BaseModel):
    id: str; title: str; authors: str; year: Optional[int] = None
    abstract: str; filename: str; extraction_confidence: Optional[float] = None

class UploadPapersResponse(BaseModel):
    session_id: str; papers: List[UploadedPaperInfo]
    total_extracted: int; failed_files: List[str]

class CitationEvalRequest(BaseModel):
    sentence: str; paper_title: str; paper_abstract: str

class SectionEvalRequest(BaseModel):
    section_text: str; topic: str
    project_context: Optional[str] = None
    citations: Optional[List[Dict]] = []

class SystemVsBaselineRequest(BaseModel):
    topic: str; rag_generated_text: str; gpt_generated_text: str

class MultiApproachRequest(BaseModel):
    topic: str
    uploaded_paper_ids: Optional[List[str]] = []
    session_id: Optional[str] = None

class GraphAnalysisRequest(BaseModel):
    """
    FIX: Added default values so 422 error no longer occurs
    when optional fields are missing.
    """
    topic: str
    project_context: Optional[str] = None
    graph_type: str = "unknown"           # ← was required, now has default
    graph_description: str = ""           # ← was required, now has default
    session_id: Optional[str] = None

class ResearchGapsRequest(BaseModel):
    topic: str; session_id: Optional[str] = None

class PaperMetricsRequest(BaseModel):
    generated_text: str
    reference_texts: Optional[List[str]] = []


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/datasets/discover", response_model=List[DatasetCard])
async def discover_datasets(
    topic: str, top_k: int = 8,
    desired_features: Optional[str] = None, domain: Optional[str] = None,
):
    if not topic or len(topic.strip()) < 3:
        raise HTTPException(400, "Topic must be at least 3 characters")
    features_list = [f.strip() for f in desired_features.split(",") if f.strip()] if desired_features else None
    try:
        datasets = dataset_engine.discover_datasets(
            topic=topic.strip(), top_k=top_k,
            desired_features=features_list,
            domain=domain.strip() if domain else None,
        )
        return [
            DatasetCard(
                id=ds.id, source=ds.source, title=ds.title, description=ds.description,
                rows=ds.rows, columns=ds.columns, size_mb=ds.size_mb, tags=ds.tags,
                relevance_score=round(ds.relevance_score, 3),
            )
            for ds in datasets
        ]
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/datasets/load", response_model=DatasetLoadResponse)
async def load_dataset(dataset_id: str = Form(...), topic: str = Form("")):
    ds: Optional[DatasetInfo] = None

    try:
        live_catalog = _get_live_catalog()
        raw_entry = next((e for e in live_catalog if e["id"] == dataset_id), None)
        if raw_entry:
            ds = _datasetinfo_from_catalog(raw_entry)
    except Exception as e:
        print(f"[load_dataset] catalog lookup failed: {e}")

    if ds is None:
        try:
            results = dataset_engine.discover_datasets(topic or dataset_id, top_k=20)
            ds = next((d for d in results if d.id == dataset_id), None)
        except Exception as e:
            print(f"[load_dataset] discover fallback failed: {e}")

    if ds is None and dataset_engine.kaggle_username and dataset_engine.kaggle_key:
        try:
            import requests as _req
            resp = _req.get(
                f"https://www.kaggle.com/api/v1/datasets/{dataset_id}",
                auth=(dataset_engine.kaggle_username, dataset_engine.kaggle_key),
                timeout=10,
            )
            if resp.status_code == 200:
                item = resp.json()
                ds = DatasetInfo(
                    id=dataset_id, source="kaggle",
                    title=item.get("title", dataset_id),
                    description=(item.get("subtitle", "") or "")[:250],
                    rows=None, columns=None,
                    size_mb=round(item.get("totalBytes", 0) / 1_000_000, 1),
                    tags=[t["name"] for t in item.get("tags", [])[:6]],
                    download_url=f"kaggle:{dataset_id}", file_name="data.csv",
                    relevance_score=1.0, match_explanation="direct Kaggle lookup",
                )
        except Exception as e:
            print(f"[load_dataset] Kaggle lookup failed: {e}")

    if ds is None:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found")

    try:
        df, metadata = dataset_engine.load_dataset(ds)
    except Exception as e:
        raise HTTPException(500, f"Failed to load dataset '{ds.title}': {e}")

    if df is None or df.empty:
        raise HTTPException(500, f"Dataset '{ds.title}' returned empty DataFrame")

    original_rows = len(df)
    sampled       = False

    if original_rows > GRAPH_SAMPLE_LIMIT:
        target_col = metadata.get("target_col")
        if target_col and target_col in df.columns and df[target_col].nunique() < 50:
            df = (
                df.groupby(target_col, group_keys=False)
                  .apply(lambda g: g.sample(
                      min(len(g), max(1, int(GRAPH_SAMPLE_LIMIT * len(g) / original_rows))),
                      random_state=42))
                  .sample(frac=1, random_state=42)
                  .reset_index(drop=True)
            )
        else:
            df = df.sample(n=GRAPH_SAMPLE_LIMIT, random_state=42).reset_index(drop=True)
        sampled  = True
        metadata = {**metadata, "rows": len(df)}

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "df": df, "metadata": metadata, "dataset_info": ds,
        "graphs": [], "graph_context": "", "uploaded_papers": [],
        "original_rows": original_rows, "sampled": sampled,
        "topic": topic, "project_context": None,
    }

    return DatasetLoadResponse(
        session_id=session_id, rows=len(df), original_rows=original_rows,
        sampled=sampled, columns=metadata["columns"],
        numeric_cols=metadata["numeric_cols"], categorical_cols=metadata["categorical_cols"],
        target_col=metadata.get("target_col"), missing_pct=metadata["missing_pct"],
        recommended_graphs=metadata["recommended_graphs"], preview=metadata["preview"],
    )


@app.get("/datasets/quality")
async def dataset_quality(session_id: str):
    session = _sessions.get(session_id)
    if not session or session.get("df") is None:
        raise HTTPException(404, "Session not found or dataset not loaded")
    try:
        return quality_report(session["df"])
    except Exception as e:
        raise HTTPException(500, f"Quality report failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/graphs/generate", response_model=List[GraphResult])
async def generate_graphs(
    session_id: str = Form(...),
    graph_types: str = Form(...),
    topic: Optional[str] = Form(None),
    project_context: Optional[str] = Form(None),
):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found — load a dataset first")

    df: Optional[pd.DataFrame] = session.get("df")
    if df is None or df.empty:
        raise HTTPException(400, "Session has no loaded dataset.")

    ds: DatasetInfo = session["dataset_info"]
    if ds is None:
        raise HTTPException(400, "Session has no dataset_info.")

    selected = [t.strip() for t in graph_types.split(",") if t.strip()]
    if not selected:
        raise HTTPException(400, "No graph types specified")

    metadata       = session["metadata"]
    resolved_topic = (topic or session.get("topic") or ds.title).strip()

    parsed_project_ctx: Optional[Dict] = None
    if project_context:
        try:
            parsed_project_ctx = json.loads(project_context)
        except Exception:
            pass
    if parsed_project_ctx is None:
        parsed_project_ctx = session.get("project_context")

    session["topic"]           = resolved_topic
    session["project_context"] = parsed_project_ctx

    try:
        graphs = graph_gen.generate(
            df=df, metadata=metadata, selected_types=selected,
            dataset_title=ds.title,
            topic=resolved_topic,
            project_context=parsed_project_ctx,
        )
    except Exception as e:
        raise HTTPException(500, f"Graph generation failed: {e}\n{traceback.format_exc()}")

    # Enrich with project_insight
    try:
        from llm_generator import _call
        for g in graphs:
            proj_ctx_block = ""
            if parsed_project_ctx:
                proj_ctx_block = (
                    f"Project: {parsed_project_ctx.get('title', resolved_topic)}\n"
                    f"Summary: {parsed_project_ctx.get('summary', '')[:200]}"
                )
            prompt = (
                f"Research topic: {resolved_topic}\n{proj_ctx_block}\n"
                f"Graph: {g['title']}\n"
                f"Statistical finding: {g.get('statistical_insight', '')}\n\n"
                "In 3-5 sentences explain: (1) what this graph reveals, "
                "(2) how the finding strengthens this research project, "
                "(3) what next step it motivates. No bullet points."
            )
            raw_insight = _call(
                "Concise academic research analyst. IEEE style. 3-5 sentences max.",
                prompt, max_tokens=220, temp=0.3,
            )
            g["project_insight"] = raw_insight.strip()
            g["insight"]         = raw_insight.strip()
    except Exception as e:
        print(f"[generate_graphs] project_insight enrichment failed: {e}")

    session["graphs"] = graphs
    session["graph_context"] = graph_gen.build_llm_context(
        graphs, ds.title, topic=resolved_topic, project_context=parsed_project_ctx,
    )

    return [
        GraphResult(
            id=g["id"], type=g["type"], title=g["title"],
            figure_label=g["figure_label"], data=g["data"],
            statistical_insight=g.get("statistical_insight", g.get("insight", "")),
            project_insight=g.get("project_insight", g.get("insight", "")),
            insight=g.get("project_insight", g.get("insight", "")),
            stats=g.get("stats", {}),
        )
        for g in graphs
    ]


@app.post("/graphs/analyze-relevance")
async def graph_relevance_endpoint(req: GraphAnalysisRequest):
    """
    FIX: 422 error was caused by missing required fields.
    Both graph_type and graph_description now have defaults.
    """
    dataset_meta: Dict = {}
    if req.session_id and req.session_id in _sessions:
        dataset_meta = _sessions[req.session_id].get("metadata") or {}
    try:
        result = analyze_graph_relevance(
            topic=req.topic,
            project_context=req.project_context,
            graph_type=req.graph_type,
            graph_description=req.graph_description,
            dataset_meta=dataset_meta,
        )
        return result
    except Exception as e:
        print(f"[graph_relevance_endpoint] Error: {e}")
        return {
            "what_it_represents": "A visualization of the dataset.",
            "key_insights":       ["Data analysis visualization."],
            "relevance_to_research": "Provides data-driven evidence.",
            "decision":           "YES",
            "justification":      "Standard analytical graph.",
            "suggested_section":  "Results",
        }


@app.post("/graphs/analyze-all-relevance")
async def graph_all_relevance_endpoint(
    session_id: str = Form(...),
    topic: str = Form(...),
    project_context: Optional[str] = Form(None),
):
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    graphs       = session.get("graphs", [])
    if not graphs:
        raise HTTPException(400, "No graphs generated yet.")
    dataset_meta = session.get("metadata") or {}
    results = []
    for g in graphs:
        analysis = analyze_graph_relevance(
            topic=topic, project_context=project_context,
            graph_type=g.get("type", "unknown"),
            graph_description=g.get("insight", g.get("title", "")),
            dataset_meta=dataset_meta,
        )
        results.append({"graph_id": g["id"], "graph_title": g["title"], "analysis": analysis})
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PAPER UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/upload-papers", response_model=UploadPapersResponse)
async def upload_papers(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
):
    if not files:
        raise HTTPException(400, "No files provided")

    tmp_dir     = tempfile.mkdtemp(prefix="user_papers_")
    saved_paths = []
    failed      = []

    for upload in files:
        ext = os.path.splitext(upload.filename)[1].lower()
        if ext not in {".pdf", ".docx", ".doc"}:
            failed.append(f"{upload.filename} (unsupported: {ext})")
            continue
        dest = os.path.join(tmp_dir, upload.filename)
        try:
            content = await upload.read()
            with open(dest, "wb") as f:
                f.write(content)
            saved_paths.append(dest)
        except Exception as e:
            failed.append(f"{upload.filename} ({e})")

    extracted = extract_papers(saved_paths, deduplicate=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if not extracted and not failed:
        raise HTTPException(422, "No valid papers could be extracted.")

    if session_id and session_id in _sessions:
        _sessions[session_id]["uploaded_papers"] = extracted
    else:
        session_id = str(uuid.uuid4())
        _sessions[session_id] = _make_empty_session()
        _sessions[session_id]["uploaded_papers"] = extracted

    return UploadPapersResponse(
        session_id=session_id,
        papers=[
            UploadedPaperInfo(
                id=p["id"], title=p["title"], authors=p["authors"], year=p.get("year"),
                abstract=p["abstract"][:300] + ("…" if len(p["abstract"]) > 300 else ""),
                filename=p["filename"],
                extraction_confidence=p.get("extraction_confidence"),
            )
            for p in extracted
        ],
        total_extracted=len(extracted),
        failed_files=failed,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH GAPS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/research-gaps")
async def research_gaps_endpoint(req: ResearchGapsRequest):
    rag_papers: List[Dict] = []
    try:
        rag        = get_rag()
        rag_papers = rag.search(req.topic, top_k=15)
    except Exception as e:
        print(f"[research-gaps] RAG failed: {e}")

    if req.session_id and req.session_id in _sessions:
        uploaded   = _sessions[req.session_id].get("uploaded_papers", [])
        rag_papers = uploaded + rag_papers

    return detect_research_gaps(req.topic, rag_papers)


# ═══════════════════════════════════════════════════════════════════════════════
# PAPER METRICS (Perplexity, BLEU, ROUGE)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/evaluate/metrics")
async def evaluate_paper_metrics(req: PaperMetricsRequest):
    """
    NEW ENDPOINT: Compute Perplexity, BLEU, ROUGE-1/2/L on generated text.
    Pass reference_texts (e.g., retrieved paper abstracts) for BLEU/ROUGE comparison.
    """
    try:
        metrics = evaluate_paper(req.generated_text, req.reference_texts or [])
        return metrics
    except Exception as e:
        raise HTTPException(500, f"Metrics computation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATE — v6 with GNN pipeline + metrics
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/generate-paper")
async def generate_paper(
    topic: str                         = Form(...),
    include_graphs: bool               = Form(True),
    num_references: int                = Form(10),
    use_custom_dataset: bool           = Form(False),
    dataset_file: Optional[UploadFile] = File(None),
    session_id: Optional[str]          = Form(None),
    use_uploaded_papers_only: bool     = Form(False),
    project_context: Optional[str]     = Form(None),
    include_explainability: bool       = Form(False),
):
    clear_llm_cache()
    graphs, graph_context, dataset_title = [], "", ""
    uploaded_papers: List[Dict] = []

    parsed_project_context: Optional[Dict] = None
    if project_context:
        try:
            parsed_project_context = json.loads(project_context)
            if parsed_project_context.get("title") and (
                not topic or topic == parsed_project_context.get("title")
            ):
                topic = parsed_project_context["title"]
        except Exception as e:
            print(f"[project_context parse warning] {e}")

    if session_id and session_id in _sessions:
        sess            = _sessions[session_id]
        graphs          = sess.get("graphs", [])
        graph_context   = sess.get("graph_context", "")
        uploaded_papers = sess.get("uploaded_papers", [])
        if sess.get("dataset_info"):
            dataset_title = sess["dataset_info"].title
        if parsed_project_context is None and sess.get("project_context"):
            parsed_project_context = sess["project_context"]

    # ── Paper retrieval ────────────────────────────────────────────────────────
    if uploaded_papers and use_uploaded_papers_only:
        retrieved_papers = uploaded_papers[:num_references]
    else:
        rag_papers: List[Dict] = []
        try:
            rag        = get_rag()
            rag_count  = max(1, num_references - len(uploaded_papers))
            rag_papers = rag.search(topic, top_k=rag_count)
        except Exception as e:
            print(f"RAG warning: {e}")
        retrieved_papers = (uploaded_papers + rag_papers)[:num_references]

    # ── Initialize LLM generator + build knowledge graph ──────────────────────
    try:
        llm = LLMPaperGenerator()
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

    # STAGE 2+3: Build knowledge graph from retrieved papers
    print(f"[generate_paper] Building knowledge graph from {len(retrieved_papers)} papers...")
    kg = llm.build_knowledge_graph(retrieved_papers)
    print(f"[generate_paper] KG: {len(kg.nodes)} nodes, {len(kg.edges)} edges")

    # ── Section generation ─────────────────────────────────────────────────────
    section_keys = [
        "abstract", "introduction", "literature_survey", "methodology",
        "algorithms", "block_diagram", "results", "conclusion",
    ]
    sections: Dict[str, str]        = {}
    section_analyses: Dict[str, Dict] = {}
    grounding_reports: Dict[str, List] = {}

    # Semantic citation manager (pure embedding similarity — NO keyword matching)
    cit_mgr = AdvancedCitationManager(CitationFormat.IEEE)
    for paper in retrieved_papers:
        cit_mgr.add_paper(paper)

    for key in section_keys:
        context = graph_context if key == "results" and graph_context else None
        try:
            if include_explainability:
                result = llm.generate_section_with_explainability(
                    section_type=key, topic=topic,
                    retrieved_papers=retrieved_papers,
                    context=context, project_context=parsed_project_context,
                    graphs=graphs if key == "results" else None,
                )
                raw_text              = result["content"]
                section_analyses[key] = result["analysis"]
            else:
                raw_text = llm.generate_section(
                    section_type=key, topic=topic,
                    retrieved_papers=retrieved_papers,
                    context=context, project_context=parsed_project_context,
                    graphs=graphs if key == "results" else None,
                )

            # ── SEMANTIC citation injection (embedding cosine, no keywords) ────
            enriched_text, report = cit_mgr.insert_citations_smart(
                raw_text, section_type=key
            )
            sections[key]          = enriched_text
            grounding_reports[key] = report

        except Exception as e:
            print(f"[generate_paper] Section '{key}' failed: {e}")
            sections[key] = f"[Section generation failed: {e}]"

    # ── Format bibliography ────────────────────────────────────────────────────
    try:
        bib      = cit_mgr.format_bibliography()
        citations = [{"id": i + 1, "text": t} for i, t in enumerate(bib)]
    except Exception:
        citations = [
            {
                "id":   i + 1,
                "text": (
                    f'[{i+1}] {p.get("authors", "Unknown")}, '
                    f'"{p.get("title", "Untitled")}," '
                    f'{p.get("journal_ref", p.get("source", "arXiv"))}, '
                    f'{p.get("year", "n.d.")}.'
                ),
            }
            for i, p in enumerate(retrieved_papers)
        ]

    # ── Compute Perplexity, BLEU, ROUGE ───────────────────────────────────────
    paper_metrics = {}
    try:
        full_text       = " ".join(sections.values())
        reference_texts = [p.get("abstract", "") for p in retrieved_papers[:5] if p.get("abstract")]
        paper_metrics   = evaluate_paper(full_text, reference_texts)
        print(f"[generate_paper] Metrics: perplexity={paper_metrics.get('perplexity')}, "
              f"BLEU={paper_metrics.get('bleu_score')}, ROUGE-1={paper_metrics.get('rouge_1_f1')}")
    except Exception as e:
        print(f"[generate_paper] Metrics computation failed: {e}")

    # ── Build DOCX ─────────────────────────────────────────────────────────────
    temp_dir    = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"paper_{uuid.uuid4().hex[:8]}.docx")
    try:
        build_ieee_docx(
            output_path=output_path, topic=topic, sections=sections,
            citations=citations, graphs=graphs, dataset_title=dataset_title,
            metrics=paper_metrics if paper_metrics else None,
        )
    except Exception as e:
        print(f"[generate_paper] DOCX build failed: {e}")
        traceback.print_exc()

    # ── Grounding summary ─────────────────────────────────────────────────────
    total_claims     = sum(len(v) for v in grounding_reports.values())
    supported_claims = sum(
        sum(1 for r in v if r.get("supported")) for v in grounding_reports.values()
    )
    grounding_pct = (
        round(100 * supported_claims / total_claims, 1) if total_claims > 0 else None
    )

    # ── Knowledge graph stats ──────────────────────────────────────────────────
    kg_stats = {
        "nodes": len(kg.nodes),
        "edges": len(kg.edges),
        "paper_nodes":   sum(1 for v in kg.nodes.values() if v["type"] == "paper"),
        "method_nodes":  sum(1 for v in kg.nodes.values() if v["type"] == "method"),
        "dataset_nodes": sum(1 for v in kg.nodes.values() if v["type"] == "dataset"),
        "similarity_edges": sum(1 for e in kg.edges if e["relation"] == "similar_to"),
    }

    response = {
        "title":    f"A Comprehensive Study on {topic}",
        "sections": sections,
        "citations": citations,
        "graphs": [
            {
                "id":                  g["id"],
                "title":               g["title"],
                "figure_label":        g["figure_label"],
                "data":                g["data"],
                "statistical_insight": g.get("statistical_insight", g.get("insight", "")),
                "project_insight":     g.get("project_insight", g.get("insight", "")),
                "insight":             g.get("project_insight", g.get("insight", "")),
            }
            for g in graphs
        ],
        "stats": {
            "total_words":          sum(len(s.split()) for s in sections.values()),
            "num_citations":        len(citations),
            "num_graphs":           len(graphs),
            "dataset":              dataset_title or "None",
            "uploaded_papers":      len(uploaded_papers),
            "rag_papers":           len(retrieved_papers) - len(uploaded_papers),
            "has_project_context":  parsed_project_context is not None,
            "explainability":       include_explainability,
            "grounding_pct":        grounding_pct,
            "supported_claims":     supported_claims,
            "total_claims_checked": total_claims,
            "knowledge_graph":      kg_stats,
        },
        "paper_metrics": paper_metrics,
        "docx_url": f"/download/{os.path.basename(output_path)}",
    }

    if include_explainability and section_analyses:
        response["section_analyses"] = section_analyses
    if grounding_reports:
        response["grounding_reports"] = grounding_reports

    return response


@app.get("/download/{filename}")
async def download(filename: str):
    path = os.path.join(tempfile.gettempdir(), filename)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/evaluate/citation")
async def evaluate_citation_endpoint(req: CitationEvalRequest):
    return evaluate_citation(req.sentence, req.paper_title, req.paper_abstract)


@app.post("/evaluate/section")
async def evaluate_section_endpoint(req: SectionEvalRequest):
    return evaluate_section_quality(
        req.section_text, req.topic, req.project_context, req.citations or [],
    )


@app.post("/evaluate/section-analysis")
async def section_analysis_endpoint(
    section_type: str = Form(...),
    section_text: str = Form(...),
    session_id: Optional[str] = Form(None),
):
    papers: List[Dict] = []
    if session_id and session_id in _sessions:
        papers = _sessions[session_id].get("uploaded_papers", [])
    if not papers:
        try:
            rag    = get_rag()
            papers = rag.search(section_type + " " + section_text[:100], top_k=6)
        except Exception:
            pass
    return generate_section_with_analysis(section_type, section_text, papers)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/compare/multi-approach")
async def multi_approach_comparison_endpoint(req: MultiApproachRequest):
    uploaded_papers: List[Dict] = []
    if req.session_id and req.session_id in _sessions:
        uploaded_papers = _sessions[req.session_id].get("uploaded_papers", [])
    rag_papers: List[Dict] = []
    try:
        rag        = get_rag()
        rag_papers = rag.search(req.topic, top_k=8)
    except Exception as e:
        print(f"RAG warning: {e}")
    return generate_multi_approach_comparison(
        topic=req.topic, uploaded_papers=uploaded_papers, rag_papers=rag_papers,
    )


@app.post("/compare/system-vs-baseline")
async def system_vs_baseline_endpoint(req: SystemVsBaselineRequest):
    return compare_system_vs_baseline(
        topic=req.topic,
        rag_generated_text=req.rag_generated_text,
        gpt_generated_text=req.gpt_generated_text,
    )


@app.get("/compare/tools")
async def compare_tools_endpoint():
    return {"comparison_text": generate_system_comparison_text()}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/session/clear-cache")
async def clear_session_cache(session_id: Optional[str] = Form(None)):
    clear_llm_cache()
    clear_duplicate_cache()
    if session_id and session_id in _sessions:
        del _sessions[session_id]
    return {"status": "cleared"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "6.0.0",
        "features": [
            "RAG (FAISS IVF-PQ)",
            "GNN Knowledge Graph",
            "Semantic Citation Injection (no keyword matching)",
            "Perplexity / BLEU / ROUGE metrics",
            "IEEE two-column DOCX",
            "Graph relevance analysis (422 fixed)",
        ]
    }


# Error handler for validation errors
@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    print(f"[422 Validation Error] {request.url}: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "url": str(request.url)},
    )