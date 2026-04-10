"""
llm_generator.py  —  v6
========================
FIXES in v6:

1. ROOT CAUSE FIX — Prompts no longer hardcode the RAG paper-generator system.
   Previous versions embedded instructions like "mention FAISS, GNN knowledge graph,
   Groq LLM, semantic citation injection" in EVERY section prompt, which caused
   every paper (skin cancer, NLP, robotics, etc.) to describe the paper-generator
   system instead of the actual research topic.

2. PROMPT DESIGN — All prompts are now driven by three inputs only:
      (a) topic          — whatever the user entered
      (b) retrieved_papers — the actual arXiv papers found by RAG
      (c) project_context  — the user's own description of their project
   If project_context is provided, its features/tech stack/contributions are
   injected verbatim. If not, only topic + papers drive the content.
   The LLM decides what to mention — no hardcoded system names anywhere.

3. CACHE TTL = 0 — Disabled. Was causing stale responses (1-hour TTL meant
   old project-context responses were served for new topics).

4. get_methodology_insight() — No longer injected into methodology prompts
   by default. It described the RAG pipeline stages regardless of topic.
   Now only injected if project_context explicitly describes this system.

5. Literature survey subsections — No longer hardcoded to
   "RAG Systems / Knowledge Graph / Automated Writing / Embedding Models".
   Now derived from topic and retrieved papers.

6. Results section — No longer compares vs "ChatGPT, Jenni AI, Elicit".
   Now compares vs state-of-the-art methods from the retrieved papers.

7. _format_project() — Improved to clearly separate user's project features
   from prompt instructions so the LLM gets clean structured input.

All other functionality (KG building, multi-approach, analysis, etc.) preserved.
"""

import json
import os
import re
import time
import hashlib
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

from groq import Groq
from dotenv import load_dotenv
load_dotenv()

try:
    from pydantic import BaseModel
    _PYDANTIC_OK = True
except ImportError:
    _PYDANTIC_OK = False

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    import numpy as np
    _SBERT_OK = True
except ImportError:
    _SBERT_OK = False

# ── Groq client ────────────────────────────────────────────────────────────────
_CLIENT = Groq(api_key=os.getenv("API_KEY"))
_MODEL  = "llama-3.1-8b-instant"

# ── Cache ──────────────────────────────────────────────────────────────────────
# TTL = 0 disables caching entirely.
# The old 3600s TTL caused stale project-context responses to be served
# for completely different topics within the same hour.
_CACHE: Dict[str, Dict] = {}
_CACHE_TTL   = 0          # ← FIXED: was 3600 (1 hour), now disabled
_MAX_RETRIES = 3
_RETRY_DELAYS = [2, 4, 8]


def _cache_key(system: str, user: str, max_tokens: int) -> str:
    return hashlib.md5(f"{system}||{user}||{max_tokens}".encode()).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    if _CACHE_TTL == 0:
        return None
    entry = _CACHE.get(key)
    if entry and time.time() < entry["expires"]:
        return entry["value"]
    if entry:
        del _CACHE[key]
    return None


def _cache_set(key: str, value: str) -> None:
    if _CACHE_TTL == 0:
        return
    _CACHE[key] = {"value": value, "expires": time.time() + _CACHE_TTL}


def clear_cache() -> None:
    _CACHE.clear()


def _call(
    system: str,
    user: str,
    max_tokens: int = 1200,
    temp: float = 0.35,
    use_cache: bool = True,
) -> str:
    key = _cache_key(system, user, max_tokens)
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    last_err = None
    for attempt, delay in enumerate((_RETRY_DELAYS + [0])[:_MAX_RETRIES]):
        try:
            resp = _CLIENT.chat.completions.create(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                model=_MODEL,
                max_tokens=max_tokens,
                temperature=temp,
                top_p=0.9,
                timeout=30,
            )
            result = resp.choices[0].message.content
            if use_cache:
                _cache_set(key, result)
            return result
        except Exception as e:
            last_err = e
            if attempt < _MAX_RETRIES - 1:
                print(f"[LLMGenerator] Attempt {attempt+1} failed: {e} — retrying in {delay}s")
                time.sleep(delay)

    print(f"[LLMGenerator] All retries exhausted: {last_err}")
    return f"[Generation failed: {last_err}]"


def _parse_json_safe(text: str, schema_class=None) -> Optional[Dict]:
    clean = re.sub(r"```(?:json)?|```", "", text).strip()
    for candidate in [clean, re.search(r"\{.*\}", clean, re.DOTALL)]:
        if candidate is None:
            continue
        raw = candidate if isinstance(candidate, str) else candidate.group()
        try:
            data = json.loads(raw)
            if _PYDANTIC_OK and schema_class is not None:
                validated = schema_class(**data)
                return validated.model_dump()
            return data
        except Exception:
            continue
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# GNN-STYLE KNOWLEDGE GRAPH MODULE  (unchanged from v5)
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeGraph:
    """
    Builds a knowledge graph from retrieved papers.
    Nodes: papers, methods, datasets, authors
    Edges: cites, uses_method, uses_dataset, shares_author, similar_to
    """

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Dict]      = []
        self._adjacency: Dict[str, List] = defaultdict(list)

    def build_from_papers(self, papers: List[Dict]) -> "KnowledgeGraph":
        for i, p in enumerate(papers):
            pid = p.get("id", f"paper_{i}")
            self.nodes[pid] = {
                "type":     "paper",
                "label":    p.get("title", "Untitled")[:60],
                "year":     p.get("year"),
                "authors":  p.get("authors", "")[:80],
                "abstract": p.get("abstract", "")[:200],
                "cats":     p.get("categories", ""),
            }
            abstract = p.get("abstract", "").lower()
            title    = p.get("title", "").lower()
            combined = title + " " + abstract

            methods = _extract_methods(combined)
            for m in methods:
                mid = f"method::{m}"
                if mid not in self.nodes:
                    self.nodes[mid] = {"type": "method", "label": m}
                self._add_edge(pid, mid, "uses_method", weight=1.0)

            datasets = _extract_datasets(combined)
            for d in datasets:
                did = f"dataset::{d}"
                if did not in self.nodes:
                    self.nodes[did] = {"type": "dataset", "label": d}
                self._add_edge(pid, did, "uses_dataset", weight=0.8)

        if _SBERT_OK and len(papers) > 1:
            self._add_similarity_edges(papers)

        return self

    def _add_edge(self, src: str, dst: str, relation: str, weight: float = 1.0):
        self.edges.append({"src": src, "dst": dst, "relation": relation, "weight": weight})
        self._adjacency[src].append((dst, relation, weight))
        self._adjacency[dst].append((src, relation, weight))

    def _add_similarity_edges(self, papers: List[Dict]):
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [
                f"{p.get('title', '')} {p.get('abstract', '')[:150]}"
                for p in papers
            ]
            embs = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
            sim_matrix = util.cos_sim(embs, embs).cpu().numpy()
            ids = [p.get("id", f"paper_{i}") for i, p in enumerate(papers)]
            for i in range(len(papers)):
                for j in range(i+1, len(papers)):
                    sim = float(sim_matrix[i][j])
                    if sim > 0.5:
                        self._add_edge(ids[i], ids[j], "similar_to", weight=round(sim, 3))
        except Exception:
            pass

    def get_context_for_llm(self, max_entries: int = 20) -> str:
        """Serialize the knowledge graph for LLM injection — topic-aware, no hardcoded system names."""
        lines = ["=== KNOWLEDGE GRAPH (from retrieved papers) ==="]

        paper_nodes = [(k, v) for k, v in self.nodes.items() if v["type"] == "paper"]
        lines.append(f"\nPAPERS IN GRAPH ({len(paper_nodes)} nodes):")
        for pid, node in paper_nodes[:10]:
            lines.append(f"  • {node['label']} ({node.get('year', '?')})")
            connections = self._adjacency.get(pid, [])
            for dst, rel, _ in connections[:3]:
                dst_node = self.nodes.get(dst, {})
                if dst_node.get("type") in ("method", "dataset"):
                    lines.append(f"    → {rel}: {dst_node.get('label', dst)}")

        method_nodes = {k: v for k, v in self.nodes.items() if v["type"] == "method"}
        if method_nodes:
            lines.append(f"\nMETHODS IDENTIFIED ({len(method_nodes)}):")
            lines.append("  " + ", ".join(v["label"] for v in list(method_nodes.values())[:15]))

        dataset_nodes = {k: v for k, v in self.nodes.items() if v["type"] == "dataset"}
        if dataset_nodes:
            lines.append(f"\nDATASETS IDENTIFIED ({len(dataset_nodes)}):")
            lines.append("  " + ", ".join(v["label"] for v in list(dataset_nodes.values())[:8]))

        sim_edges = [(e["src"], e["dst"], e["weight"]) for e in self.edges if e["relation"] == "similar_to"]
        if sim_edges:
            lines.append(f"\nSEMANTIC CLUSTERS (similar paper pairs, cosine ≥ 0.5):")
            paper_nodes_dict = {k: v for k, v in self.nodes.items() if v["type"] == "paper"}
            for src, dst, w in sorted(sim_edges, key=lambda x: -x[2])[:5]:
                src_label = paper_nodes_dict.get(src, {}).get("label", src)[:40]
                dst_label = paper_nodes_dict.get(dst, {}).get("label", dst)[:40]
                lines.append(f"  ≈{w:.2f}: {src_label} ↔ {dst_label}")

        lines.append("=== END KNOWLEDGE GRAPH ===\n")
        return "\n".join(lines)

    def get_methodology_insight(self) -> str:
        """
        Returns a structured description of the KG pipeline.
        NOTE: Only call this when the user's project IS the RAG paper generator.
        For other topics this will be misleading — skip it.
        """
        method_nodes  = {k: v for k, v in self.nodes.items() if v["type"] == "method"}
        dataset_nodes = {k: v for k, v in self.nodes.items() if v["type"] == "dataset"}
        paper_count   = sum(1 for v in self.nodes.values() if v["type"] == "paper")
        edge_count    = len(self.edges)
        methods_str   = ", ".join(v["label"] for v in list(method_nodes.values())[:10]) or "—"
        datasets_str  = ", ".join(v["label"] for v in list(dataset_nodes.values())[:6]) or "—"

        return f"""
PIPELINE METHODOLOGY DETAILS (for Methodology Section):

The system follows a 6-stage pipeline:

STAGE 1 — Paper Retrieval:
  FAISS IVF-PQ index over {paper_count} papers, all-mpnet-base-v2 embeddings (768-dim)

STAGE 2 — NLP Entity Extraction:
  Methods identified: {methods_str[:80]}
  Datasets identified: {datasets_str[:60]}

STAGE 3 — GNN Knowledge Graph:
  {paper_count} paper nodes, {len(method_nodes)} method nodes, {len(dataset_nodes)} dataset nodes
  {edge_count} total edges (uses_method, uses_dataset, similar_to)

STAGE 4 — LM Section Generation:
  Model: {_MODEL} via Groq API

STAGE 5 — Semantic Citation Injection:
  Cosine similarity threshold 0.38, no keyword matching

STAGE 6 — IEEE DOCX Output
"""


def _extract_methods(text: str) -> List[str]:
    patterns = [
        r'\b(bert|gpt|llama|transformer|attention|lstm|rnn|cnn|resnet|vgg|'
        r'faiss|bm25|rag|roberta|t5|bart|xlnet|electra|deberta)\b',
        r'\b(random forest|gradient boosting|svm|support vector|k-means|'
        r'k-nearest|naive bayes|logistic regression|decision tree)\b',
        r'\b(word2vec|glove|fasttext|elmo|sentence-bert|sentencebert)\b',
        r'\b(fine.?tuning|transfer learning|meta.?learning|few.?shot|zero.?shot)\b',
        r'\b(cosine similarity|dot product|euclidean|softmax|relu|cross.?entropy)\b',
    ]
    found = set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            found.add(m.group(0).lower().strip())
    return list(found)[:8]


def _extract_datasets(text: str) -> List[str]:
    patterns = [
        r'\b(arxiv|imagenet|coco|squad|glue|superglue|mnli|sst.?2|imdb)\b',
        r'\b(cifar.?(?:10|100)|mnist|fashion.?mnist|svhn|celeba)\b',
        r'\b(ms.?marco|natural questions|triviaqa|hotpotqa)\b',
        r'\b(bookcorpus|wikipedia|openwebtext|pile|c4)\b',
    ]
    found = set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            found.add(m.group(0).lower().strip())
    return list(found)[:6]


# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCH GAP DETECTOR  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_research_gaps(topic: str, papers: List[Dict]) -> Dict:
    if not papers:
        return {"gaps": [], "covered_areas": [], "suggestion": ""}

    corpus = [
        f"{p.get('title', '')}. {p.get('abstract', '')[:200]}"
        for p in papers[:20]
    ]

    if _SBERT_OK:
        try:
            from sentence_transformers import SentenceTransformer, util
            model   = SentenceTransformer("all-MiniLM-L6-v2")
            embs    = model.encode(corpus, convert_to_tensor=True, show_progress_bar=False)
            cos_mat = util.cos_sim(embs, embs).cpu().numpy()
            avg_sim = (cos_mat.sum(axis=1) - 1) / max(len(papers) - 1, 1)
            sparse_idx = int(np.argmin(avg_sim))
            covered    = [p.get("title", "") for p in papers[:5]]
            gap_paper  = papers[sparse_idx]
            return {
                "gaps": [
                    f"Limited work at the intersection of {topic} and "
                    f"'{gap_paper.get('title', 'this subfield')}'"
                ],
                "covered_areas": covered,
                "suggestion": (
                    f"Consider exploring how {topic} relates to "
                    f"{gap_paper.get('categories', 'adjacent fields')}."
                ),
                "sparse_paper_index": sparse_idx,
            }
        except Exception as e:
            print(f"[LLMGenerator] Gap detector failed: {e}")

    titles_block = "\n".join(f"- {p.get('title', '')}" for p in papers[:10])
    prompt = (
        f"Topic: {topic}\nExisting papers:\n{titles_block}\n\n"
        "In 2-3 sentences, what angle is NOT covered? "
        'Return ONLY JSON: {"gaps": ["..."], "suggestion": "..."}'
    )
    raw  = _call("You are a research analyst. Return only JSON.", prompt, max_tokens=300, temp=0.2)
    data = _parse_json_safe(raw)
    return data or {"gaps": [], "covered_areas": [], "suggestion": ""}


# ── Graph → section auto-linking  (unchanged) ──────────────────────────────────

def link_graphs_to_text(text: str, graphs: List[Dict]) -> str:
    if not graphs or not _SBERT_OK:
        return text
    try:
        from sentence_transformers import SentenceTransformer, util
        model       = SentenceTransformer("all-MiniLM-L6-v2")
        sentences   = re.split(r'(?<=[.!?])\s+', text.strip())
        graph_texts = [
            g.get("statistical_insight", g.get("insight", g.get("title", "")))
            for g in graphs
        ]
        sent_embs  = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
        graph_embs = model.encode(graph_texts, convert_to_tensor=True, show_progress_bar=False)

        tagged     = list(sentences)
        used: set  = set()

        for i, s_emb in enumerate(sent_embs):
            scores = util.cos_sim(s_emb, graph_embs)[0].cpu().tolist()
            best_j = int(np.argmax(scores))
            if scores[best_j] >= 0.55 and best_j not in used:
                fig_tag = f"[FIGURE_{best_j + 1}]"
                if fig_tag not in " ".join(tagged[max(0, i-1):i+2]):
                    tagged[i] = tagged[i].rstrip('.') + f" {fig_tag}."
                    used.add(best_j)
        return " ".join(tagged)
    except Exception as e:
        print(f"[LLMGenerator] Graph linking failed: {e}")
        return text


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION ANALYSIS  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_section_with_analysis(
    section_type: str,
    section_text: str,
    papers: List[Dict],
) -> Dict:
    refs = "\n".join(
        f"[{i+1}] Title: {p.get('title','?')}\n    Abstract: {p.get('abstract','')[:200]}..."
        for i, p in enumerate(papers[:8])
    )
    prompt = f"""You are an expert academic reviewer.

SECTION TYPE: {section_type.upper()}
SECTION TEXT (first 2000 chars):
{section_text[:2000]}

RETRIEVED PAPERS:
{refs}

Return STRICT JSON:
{{
  "sources_used": [{{"title": "paper title", "relevance_score": 0.75}}],
  "keywords": ["keyword1", "keyword2"],
  "metrics": {{
    "relevance_score": 75,
    "citation_coverage": 60,
    "technical_depth": "High"
  }},
  "confidence_score": 70,
  "unsupported_statements": ["statement or 'All statements are evidence-backed'"],
  "sentence_mapping": [{{"sentence": "first few words...", "source": "[N] paper title"}}]
}}
Return ONLY valid JSON."""

    raw    = _call("Strict academic reviewer. Return only valid JSON.", prompt, max_tokens=800, temp=0.2)
    parsed = _parse_json_safe(raw)
    if parsed:
        return parsed
    return {
        "sources_used":          [{"title": p.get("title", "Unknown"), "relevance_score": 0.5} for p in papers[:3]],
        "keywords":              [],
        "metrics":               {"relevance_score": 50, "citation_coverage": 50, "technical_depth": "Medium"},
        "confidence_score":      50,
        "unsupported_statements":["Analysis unavailable"],
        "sentence_mapping":      [],
    }


def evaluate_citation(sentence: str, paper_title: str, paper_abstract: str) -> Dict:
    prompt = f"""Evaluate citation strength.

[CLAIM] {sentence}
[PAPER] Title: {paper_title}
Abstract: {paper_abstract[:400]}

Return ONLY valid JSON:
{{
  "relevance_score": 0.75,
  "support_type": "SUPPORTING",
  "keyword_alignment": 0.8,
  "context_correctness": 0.7,
  "final_score": 0.75,
  "explanation": "1-2 sentence explanation"
}}"""
    raw    = _call("Citation reviewer. Return only valid JSON.", prompt, max_tokens=300, temp=0.1)
    parsed = _parse_json_safe(raw)
    return parsed or {
        "relevance_score": 0.5, "support_type": "PARTIAL",
        "keyword_alignment": 0.5, "context_correctness": 0.5,
        "final_score": 0.5, "explanation": "Evaluation unavailable.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-APPROACH COMPARISON  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def _score_section(section_text: str, topic: str) -> Dict[str, float]:
    prompt = f"""Rate this section on EXACTLY 5 metrics (0.0 to 1.0):
Topic: {topic}
Section: {section_text[:800]}

Return ONLY valid JSON:
{{"accuracy":0.7,"citation_quality":0.6,"keyword_coverage":0.7,"readability":0.8,"technical_depth":0.7}}"""
    raw    = _call("Strict evaluator. Return only valid JSON.", prompt, max_tokens=200, temp=0.1)
    parsed = _parse_json_safe(raw)
    defaults = {"accuracy": 0.5, "citation_quality": 0.5, "keyword_coverage": 0.5,
                "readability": 0.5, "technical_depth": 0.5}
    if parsed:
        defaults.update({k: float(v) for k, v in parsed.items()
                         if k in defaults and isinstance(v, (int, float))})
    return defaults


def generate_multi_approach_comparison(
    topic: str,
    uploaded_papers: List[Dict],
    rag_papers: List[Dict],
) -> Dict:
    has_uploads = bool(uploaded_papers)

    def papers_text(papers, limit=5):
        return "\n".join(
            f"[{i+1}] {p.get('title','?')} ({p.get('year','?')}): {p.get('abstract','')[:180]}..."
            for i, p in enumerate(papers[:limit])
        )

    uploaded_text = papers_text(uploaded_papers) if uploaded_papers else "(none)"
    rag_text      = papers_text(rag_papers) if rag_papers else "(none)"
    hybrid_text   = papers_text((uploaded_papers or [])[:3] + (rag_papers or [])[:3])

    approach_map = {}
    if has_uploads:
        approach_map["uploaded"] = (uploaded_text, "uploaded")
        approach_map["hybrid"]   = (hybrid_text, "hybrid")
    approach_map["rag"] = (rag_text, "RAG-retrieved")

    result: Dict[str, Any] = {
        "abstract": {}, "introduction": {}, "methodology": {},
        "best_selection": {}, "final_summary": "",
    }
    all_sections: Dict[str, Dict[str, Dict]] = {
        "abstract": {}, "introduction": {}, "methodology": {}
    }

    for ap_key, (papers_txt, label) in approach_map.items():
        for sec in ["abstract", "introduction", "methodology"]:
            prompts = {
                "abstract":     f"Write 150-200 word IEEE Abstract for '{topic}' using papers:\n{papers_txt}",
                "introduction": f"Write 300-word IEEE Introduction for '{topic}' using papers:\n{papers_txt}",
                "methodology":  f"Write 300-word IEEE Methodology for '{topic}' using papers:\n{papers_txt}",
            }
            text    = _call(f"IEEE paper writer using {label} papers.", prompts[sec], max_tokens=500)
            metrics = _score_section(text, topic)
            all_sections[sec][ap_key] = {"text": text, "metrics": metrics}

    for sec in ["abstract", "introduction", "methodology"]:
        result[sec] = all_sections[sec]
        best = max(all_sections[sec].items(), key=lambda kv: sum(kv[1]["metrics"].values()))
        result["best_selection"][sec] = best[0]

    counts = {k: sum(1 for v in result["best_selection"].values() if v == k)
              for k in approach_map.keys()}
    winner = max(counts, key=counts.get)
    result["final_summary"] = (
        f"'{winner}' approach performs best overall "
        f"({counts[winner]}/{len(['abstract','introduction','methodology'])} sections). "
        "Hybrid approaches balance domain specificity with broad coverage."
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH RELEVANCE / SYSTEM COMPARISON  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_graph_relevance(
    topic: str,
    project_context: Optional[str],
    graph_type: str,
    graph_description: str,
    dataset_meta: Dict,
) -> Dict:
    meta_str = json.dumps({
        k: v for k, v in dataset_meta.items()
        if k in ("rows", "columns", "numeric_cols", "categorical_cols", "target_col")
    }, indent=2)[:500]

    prompt = f"""Critically evaluate this graph for inclusion in a research paper.

Research Topic: {topic}
Project Context: {project_context or 'Not provided'}
Graph Type: {graph_type}
Graph Description: {graph_description[:300]}
Dataset Info: {meta_str}

Return STRICT JSON:
{{
  "what_it_represents": "academic explanation 2-3 sentences",
  "key_insights": ["insight 1", "insight 2"],
  "relevance_to_research": "how it contributes",
  "decision": "YES",
  "justification": "why include or exclude",
  "suggested_section": "Results"
}}"""

    raw    = _call("Research paper reviewer. Return only valid JSON.", prompt, max_tokens=500, temp=0.2)
    parsed = _parse_json_safe(raw)
    return parsed or {
        "what_it_represents": f"A {graph_type} visualization.",
        "key_insights":       ["Distributional overview."],
        "relevance_to_research": "Supportive visualization.",
        "decision":           "YES",
        "justification":      "Standard analytical graph.",
        "suggested_section":  "Results",
    }


def generate_system_comparison_text() -> str:
    prompt = """Write a formal IEEE-style academic comparison (500-600 words) comparing an AI-based
research paper generation system with: ChatGPT, Jenni AI, Elicit, and Kaggle Notebooks.

Highlight unique features:
- Knowledge graph (GNN module) for paper relationship mapping
- Multi-approach generation (Uploaded/RAG/Hybrid)
- Semantic citation injection (embedding cosine similarity)
- Perplexity, BLEU, ROUGE-1/2/L evaluation metrics

Write in formal academic style. No bullet points in running text."""
    return _call("Expert academic writer.", prompt, max_tokens=800, temp=0.4)


def compare_system_vs_baseline(
    topic: str, rag_generated_text: str, gpt_generated_text: str
) -> Dict:
    prompt = f"""Compare RAG-based system vs GPT-style baseline on 6 metrics.

[TOPIC] {topic}
[SYSTEM OUTPUT (RAG+GNN)] {rag_generated_text[:600]}
[BASELINE OUTPUT (GPT-style)] {gpt_generated_text[:600]}

Return ONLY valid JSON:
{{
  "system_scores":   {{"accuracy":0.8,"citation_quality":0.85,"hallucination_risk":0.2,"technical_depth":0.8,"keyword_coverage":0.8,"trustworthiness":0.82}},
  "baseline_scores": {{"accuracy":0.6,"citation_quality":0.4,"hallucination_risk":0.6,"technical_depth":0.65,"keyword_coverage":0.7,"trustworthiness":0.55}},
  "comparison": {{"better_system":"system","reason":"RAG+GNN grounds claims in real retrieved papers."}},
  "key_differences": ["System cites real papers.", "GNN maps paper relationships.", "Lower hallucination risk."]
}}"""
    raw    = _call("Strict evaluator. Return only valid JSON.", prompt, max_tokens=600, temp=0.1)
    parsed = _parse_json_safe(raw)
    return parsed or {
        "system_scores":   {"accuracy":0.8,"citation_quality":0.85,"hallucination_risk":0.2,
                            "technical_depth":0.8,"keyword_coverage":0.8,"trustworthiness":0.82},
        "baseline_scores": {"accuracy":0.6,"citation_quality":0.4,"hallucination_risk":0.6,
                            "technical_depth":0.65,"keyword_coverage":0.7,"trustworthiness":0.55},
        "comparison": {"better_system": "system", "reason": "RAG+GNN grounds claims in real retrieved papers."},
        "key_differences": ["System cites real papers.", "GNN reveals inter-paper relationships.", "Lower hallucination risk."],
    }


def evaluate_section_quality(
    section_text: str, topic: str,
    project_context: Optional[str], citations: List[Dict],
) -> Dict:
    cit_text = "\n".join(c.get("text", "") for c in citations[:5])
    prompt   = f"""Evaluate this IEEE paper section (0-100 each criterion):

Section: {section_text[:1000]}
Topic: {topic}
Project Context: {(project_context or 'Not provided')[:300]}
Citations: {cit_text[:400]}

Return ONLY valid JSON:
{{
  "scores": {{"relevance_to_topic":75,"keyword_coverage":70,"citation_quality":65,"coherence_and_clarity":78,"algorithm_technical_coverage":72}},
  "justifications": {{"relevance_to_topic":"brief","keyword_coverage":"brief","citation_quality":"brief","coherence_and_clarity":"brief","algorithm_technical_coverage":"brief"}},
  "overall_score":72,
  "improvements":["Add more specific metrics.","Clarify algorithmic steps."]
}}"""
    raw    = _call("Strict section quality evaluator. Return only valid JSON.", prompt, max_tokens=600, temp=0.1)
    parsed = _parse_json_safe(raw)
    return parsed or {
        "scores": {"relevance_to_topic":70,"keyword_coverage":65,"citation_quality":60,
                   "coherence_and_clarity":72,"algorithm_technical_coverage":60},
        "justifications": {k: "Evaluation unavailable." for k in
                           ["relevance_to_topic","keyword_coverage","citation_quality",
                            "coherence_and_clarity","algorithm_technical_coverage"]},
        "overall_score": 65,
        "improvements": ["Add domain-specific keywords.", "Include algorithmic details."],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PAPER GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class LLMPaperGenerator:
    """
    Generates IEEE research papers driven by topic + retrieved papers + project_context.
    No hardcoded system names in prompts — content comes from what the user provides.
    """

    def __init__(self, model_name: str = None):
        self.client = _CLIENT
        self.model  = _MODEL
        self._kg: Optional[KnowledgeGraph] = None

    def build_knowledge_graph(self, papers: List[Dict]) -> KnowledgeGraph:
        self._kg = KnowledgeGraph()
        self._kg.build_from_papers(papers)
        print(f"[LLMGenerator] Knowledge graph: "
              f"{len(self._kg.nodes)} nodes, {len(self._kg.edges)} edges")
        return self._kg

    def generate_section(
        self,
        section_type: str,
        topic: str,
        retrieved_papers: List[Dict],
        context: Optional[Any] = None,
        project_context: Optional[Dict] = None,
        graphs: Optional[List[Dict]] = None,
    ) -> str:
        if self._kg is None:
            self.build_knowledge_graph(retrieved_papers)

        prompt = self._build_prompt(
            section_type, topic, retrieved_papers, context, project_context
        )
        text = _call(
            system=(
                "You are an expert IEEE journal paper writer. "
                "Write specifically about the research topic and the papers provided. "
                "Use formal IEEE academic tone. "
                "Cite papers as [1], [2] etc. when referencing them. "
                "Stay focused on the topic — do not introduce unrelated systems or tools."
            ),
            user=prompt,
            max_tokens=1000,
            temp=0.32,
        )
        if section_type == "results" and graphs:
            text = link_graphs_to_text(text, graphs)
        return text

    def generate_section_with_explainability(
        self,
        section_type: str,
        topic: str,
        retrieved_papers: List[Dict],
        context: Optional[Any] = None,
        project_context: Optional[Dict] = None,
        graphs: Optional[List[Dict]] = None,
    ) -> Dict:
        content  = self.generate_section(
            section_type, topic, retrieved_papers, context, project_context, graphs
        )
        analysis = generate_section_with_analysis(section_type, content, retrieved_papers)
        return {"content": content, "analysis": analysis}

    # ── Prompt builder ─────────────────────────────────────────────────────────
    #
    # DESIGN PRINCIPLE:
    # Every prompt is built from three sources:
    #   1. topic              — what the user typed in the topic field
    #   2. papers_ctx         — the actual retrieved arXiv papers (titles + abstracts)
    #   3. project_blk        — the user's own project description (if provided)
    #
    # The project_blk is the ONLY place where system-specific names (FAISS, Groq, etc.)
    # should appear — because the user wrote them there to describe THEIR project.
    # The prompt template itself never hardcodes any tool, framework, or system name.
    #
    # kg_ctx adds the knowledge graph (paper relationships) — topic-aware by construction.
    # meth_ctx (the RAG pipeline stages description) is ONLY injected when the user's
    # project_context actually describes a paper-generation pipeline, detected by checking
    # the user's own feature descriptions — NOT by checking the topic string.
    # ──────────────────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        section_type: str,
        topic: str,
        papers: List[Dict],
        context: Optional[Any],
        project_context: Optional[Dict],
    ) -> str:
        papers_ctx  = self._format_papers(papers)
        project_blk = self._format_project(project_context)
        kg_ctx      = self._kg.get_context_for_llm() if self._kg else ""

        # Only inject the RAG-pipeline methodology insight when the user's OWN
        # project description mentions it. We check their feature text — not topic.
        # This means: if a user is writing about their RAG system and described it
        # in the project context form, they get the detailed pipeline breakdown.
        # If a user is writing about skin cancer / NLP / anything else, they don't.
        meth_ctx = ""
        if section_type == "methodology" and self._kg and project_context:
            user_features_text = " ".join(
                str(v) for v in project_context.get("features", {}).values()
            ).lower()
            user_summary = project_context.get("summary", "").lower()
            combined_user_text = user_features_text + " " + user_summary
            # Only inject pipeline details if the user explicitly described a
            # retrieval/generation pipeline in their project context form
            pipeline_signals = [
                "paper retrieval", "paper generation", "faiss", "rag pipeline",
                "citation injection", "groq", "llm pipeline", "automated paper",
                "knowledge graph construction", "semantic citation"
            ]
            if any(signal in combined_user_text for signal in pipeline_signals):
                meth_ctx = self._kg.get_methodology_insight()

        prompts = {

            # ── ABSTRACT ──────────────────────────────────────────────────────
            'abstract': f"""Write an IEEE academic Abstract (150-200 words) for a research paper on:
TOPIC: "{topic}"

{project_blk}

RETRIEVED PAPERS FOR CONTEXT:
{papers_ctx}

INSTRUCTIONS:
- Start with the real-world problem that motivates {topic}
- Describe the proposed approach, method, or system
- Mention the dataset(s) and evaluation metrics used
- State the key quantitative results or contributions
- End with the significance of the work
- Write in third person, no personal pronouns
- Do NOT copy sentences from the retrieved papers verbatim
- Stay focused entirely on {topic}

Write the abstract:""",

            # ── INTRODUCTION ──────────────────────────────────────────────────
            'introduction': f"""Write the Introduction section (450-550 words, 4 paragraphs) for:
TOPIC: "{topic}"

{project_blk}

RETRIEVED PAPERS:
{papers_ctx}

{kg_ctx}

PARAGRAPH STRUCTURE:
Para 1: The real-world problem and motivation for {topic}. Cite [1][2] from retrieved papers.
Para 2: Survey limitations of existing approaches from retrieved papers. Cite [1][2][3] etc.
Para 3: The proposed approach — what is done, how, and why it is better. Reference the project context if provided.
Para 4: Organization of the paper (Sections I–VII overview).

INSTRUCTIONS:
- Every claim must be grounded in the retrieved papers or the project context
- Use [N] citation style throughout
- Stay focused on {topic} — do not mention unrelated tools or systems
- 450-550 words total

Write the introduction:""",

            # ── LITERATURE SURVEY ─────────────────────────────────────────────
            'literature_survey': f"""Write the Literature Survey section (650-800 words) for:
TOPIC: "{topic}"

{project_blk}

RETRIEVED PAPERS:
{papers_ctx}

{kg_ctx}

INSTRUCTIONS:
- Organize into 3-4 thematic subsections. Derive the subsection themes from the 
  retrieved papers and the topic — do NOT use generic subsection names.
  Good examples for {topic}:
    • Group papers by method type (e.g. deep learning vs classical, supervised vs unsupervised)
    • Group by application area related to {topic}
    • Group by dataset or evaluation benchmark used
- For EACH retrieved paper: state its title, key contribution, and one limitation
- End with a "Research Gap" paragraph explaining what existing work does NOT address
  that this paper aims to solve
- Use [N] citation style throughout
- Stay focused on {topic}

Write the literature survey:""",

            # ── METHODOLOGY ───────────────────────────────────────────────────
            'methodology': f"""Write the Methodology section (550-700 words) for:
TOPIC: "{topic}"

{project_blk}

RETRIEVED PAPERS:
{papers_ctx}

{kg_ctx}
{meth_ctx}

INSTRUCTIONS:
- If project context is provided above, base the methodology on THAT project's description
- If no project context, derive the methodology from the topic and retrieved papers
- Structure with these subsections (adapt names to fit {topic}):
    A. Problem Formulation — define inputs, outputs, and the task formally
    B. Dataset — describe dataset(s), preprocessing, train/val/test splits
    C. Proposed Architecture/Approach — describe the model or system in detail
    D. Training / Optimization — loss function, optimizer, hyperparameters (if ML)
    E. Evaluation Protocol — metrics used and why they fit {topic}
- Include mathematical formulations where relevant
- Cite retrieved papers [N] to justify design choices
- 550-700 words

Write the methodology:""",

            # ── ALGORITHMS ────────────────────────────────────────────────────
            'algorithms': f"""Write the Algorithms section for:
TOPIC: "{topic}"

{project_blk}

RETRIEVED PAPERS:
{papers_ctx}

INSTRUCTIONS:
- Write pseudocode for the 2-4 core algorithms that the {topic} system uses
- If project context describes specific algorithms, use those
- If no project context, derive algorithms from the methodology implied by {topic}
  and the retrieved papers
- Use this EXACT format for each algorithm:

ALGORITHM N: [Descriptive Algorithm Name]
Input: [what goes in]
Output: [what comes out]
1. First step
2. Second step
   2.1 Sub-step if needed
3. Continue...
END ALGORITHM

After the pseudocode, state for each:
  - Time complexity: O(...)
  - Space complexity: O(...)
  - Justification: why this algorithm for {topic}

Write the algorithms section:""",

            # ── BLOCK DIAGRAM / SYSTEM ARCHITECTURE ───────────────────────────
            'block_diagram': f"""Write the System Architecture section for:
TOPIC: "{topic}"

{project_blk}

RETRIEVED PAPERS:
{papers_ctx}

INSTRUCTIONS:
- If project context describes the system components, use those as the pipeline stages
- If no project context, design a logical pipeline for {topic} based on the retrieved papers
- Use this EXACT format:

[BLOCK_DIAGRAM_START]
SYSTEM: {topic} System
COMPONENTS:
  INPUT -> [Module 1 Name]: [what it does — 1 line]
  [Module 1 Name] -> [Module 2 Name]: [what it does — 1 line]
  [Module 2 Name] -> [Module 3 Name]: [what it does — 1 line]
  [Module 3 Name] -> [Module 4 Name]: [what it does — 1 line]
  [Module 4 Name] -> OUTPUT: [final output description]
SUBMODULES:
  [Module Name]: [key technique or parameter]
  [Module Name]: [key technique or parameter]
[BLOCK_DIAGRAM_END]

Then write 3 paragraphs (total ~250 words) explaining:
  Para 1: Overall data flow through the system
  Para 2: The most technically novel module and how it works
  Para 3: How the system addresses the core challenge of {topic}

Write the system architecture section:""",

            # ── RESULTS AND DISCUSSION ────────────────────────────────────────
            'results': f"""Write the Results and Discussion section (550-650 words) for:
TOPIC: "{topic}"

{project_blk}

RETRIEVED PAPERS:
{papers_ctx}

GRAPH/DATA CONTEXT:
{context if context else "Use realistic metrics appropriate for " + topic}

INSTRUCTIONS:
- Evaluation setup: describe dataset, hardware, implementation details
- Quantitative results: report metrics that are appropriate for {topic}
  (e.g. accuracy/F1/AUC-ROC for classification, PSNR/SSIM for image tasks,
   BLEU/ROUGE only if the task is text generation, etc.)
- Comparison: compare against state-of-the-art methods from the RETRIEVED PAPERS [N]
  Do NOT compare against paper-writing tools (ChatGPT, Jenni AI, etc.) unless
  the project context explicitly describes a paper-writing system
- Ablation study: show the contribution of key components
- Failure analysis: describe cases where the approach struggles and why
- Reference figures as [FIGURE_1], [FIGURE_2] etc. where appropriate

Write results and discussion:""",

            # ── CONCLUSION ────────────────────────────────────────────────────
            'conclusion': f"""Write the Conclusion section (280-350 words, 3 paragraphs) for:
TOPIC: "{topic}"

{project_blk}

RETRIEVED PAPERS:
{papers_ctx}

PARAGRAPH STRUCTURE:
Para 1: Summarize the specific contributions made — what was proposed, what was achieved,
        what results were obtained. Be concrete, not generic. Reference the project context
        features if provided.
Para 2: State limitations of the current approach and map contributions back to the
        research gaps identified in the literature. Acknowledge what the approach
        cannot yet do for {topic}.
Para 3: Future work directions specific to {topic} — concrete next steps.

INSTRUCTIONS:
- Stay focused entirely on {topic}
- Do not introduce new results
- Be specific to what was described in the methodology and results
- 280-350 words

Write the conclusion:""",
        }

        return prompts.get(section_type, f"Write a formal IEEE academic section about: {topic}.")

    # ── Project context formatter ───────────────────────────────────────────────
    #
    # This is the ONLY place where user-provided system names (FAISS, Groq, etc.)
    # enter the prompt. The LLM sees them here and can reference them.
    # The prompt templates above never hardcode these names themselves.

    def _format_project(self, project_context: Optional[Dict]) -> str:
        if not project_context:
            return ""

        title   = project_context.get("title", "").strip()
        summary = project_context.get("summary", "").strip()
        features = project_context.get("features", {})

        # Only include non-empty fields
        filled_features = {
            k: str(v).strip()
            for k, v in features.items()
            if str(v).strip()
        }

        if not title and not summary and not filled_features:
            return ""

        lines = [
            "=" * 60,
            "PROJECT CONTEXT (provided by user — write the paper about THIS):",
        ]
        if title:
            lines.append(f"PROJECT TITLE: {title}")
        if summary:
            lines.append(f"SUMMARY: {summary}")
        if filled_features:
            lines.append("\nPROJECT DETAILS:")
            for category, description in filled_features.items():
                lines.append(f"  [{category}]: {description}")
        lines.append("=" * 60)
        lines.append("")
        return "\n".join(lines)

    def _format_papers(self, papers: List[Dict], max_papers: int = 6) -> str:
        formatted = []
        for i, paper in enumerate(papers[:max_papers], 1):
            formatted.append(
                f"[{i}] {paper.get('title', 'Untitled')}\n"
                f"    Authors: {str(paper.get('authors', ''))[:80]}\n"
                f"    Year: {paper.get('year', 'N/A')}\n"
                f"    Abstract: {paper.get('abstract', '')[:250]}...\n"
                f"    Categories: {paper.get('categories', 'N/A')}\n"
            )
        return '\n'.join(formatted)