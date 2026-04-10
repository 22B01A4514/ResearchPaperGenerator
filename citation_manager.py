"""
citation_manager.py  —  v4  (Semantic-Only Edition)
=====================================================
MAJOR OVERHAUL in v4:

  SOLUTION — Pure Semantic Embedding Injection:
    1. Every paper is encoded with SentenceTransformer (all-mpnet-base-v2,
       768-dim — same model as the RAG engine for consistency).
    2. Every sentence in the generated text is encoded.
    3. Citation is injected when cosine_sim(sentence, paper) ≥ threshold (0.38).
    4. No regex triggers, no keyword lists, no hardcoded domain terms.
    5. If SBERT unavailable → BM25 with Okapi-B25 (still better than regex).

  Additional fixes:
    - 422 error on /graphs/analyze-relevance fixed in main.py (separate file).
    - Bibliography fallback always produces clean IEEE format.
    - Density targets tuned per section (abstract=0, results=0.3, etc.)
    - Duplicate injection prevention per sentence index.
    - Crossref DOI enrichment preserved.

  Metrics added (BLEU, ROUGE, Perplexity) — see evaluate_paper() below.
"""

import re
import math
import requests
import hashlib
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime
from collections import Counter


# ── Optional heavy deps ────────────────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    import numpy as np
    _SBERT_OK = True
except ImportError:
    _SBERT_OK = False
    np = None

try:
    from rank_bm25 import BM25Okapi
    _BM25_OK = True
except ImportError:
    _BM25_OK = False

try:
    import citeproc
    from citeproc import CitationStylesStyle, CitationStylesBibliography
    from citeproc import Citation, CitationItem
    from citeproc.source.json import CiteProcJSON
    import citeproc.formatter.plain as _plain_formatter
    _CITEPROC_OK = True
except ImportError:
    _CITEPROC_OK = False


class CitationFormat(Enum):
    IEEE      = "ieee"
    APA       = "apa"
    MLA       = "modern-language-association"
    CHICAGO   = "chicago-author-date"
    HARVARD   = "harvard-cite-them-right"
    VANCOUVER = "vancouver"
    BIBTEX    = "bibtex"


# ── Per-section citation density targets ──────────────────────────────────────
# 0 = never inject citations in this section
_DENSITY_TARGETS = {
    "abstract":          0,      # never cite in abstract
    "introduction":      0.30,
    "literature_survey": 0.55,
    "methodology":       0.35,
    "algorithms":        0.20,
    "block_diagram":     0.10,
    "results":           0.30,
    "conclusion":        0.20,
    "default":           0.25,
}

# Semantic similarity threshold — sentences above this get citations
_SEM_THRESHOLD    = 0.38   # tuned: low enough to catch paraphrases, high enough to avoid noise
_SUPPORT_THRESHOLD = 0.42  # sentence considered "grounded" above this


# ── Shared model instance (loaded once) ──────────────────────────────────────
_SHARED_MODEL: Optional[object] = None

def _get_model():
    global _SHARED_MODEL
    if _SHARED_MODEL is None and _SBERT_OK:
        try:
            # Use same model as RAG engine for embedding consistency
            _SHARED_MODEL = SentenceTransformer("all-mpnet-base-v2")
        except Exception:
            try:
                _SHARED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                pass
    return _SHARED_MODEL


def _load_csl_style(style_name: str):
    if not _CITEPROC_OK:
        return None
    for name in [style_name, "ieee"]:
        try:
            style = CitationStylesStyle(name, validate=False)
            return style
        except Exception:
            continue
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SEMANTIC PAPER INDEX
# ═══════════════════════════════════════════════════════════════════════════════

class _SemanticPaperIndex:
    """
    Builds embeddings for all retrieved papers.
    Uses cosine similarity (no keyword matching whatsoever).
    """

    def __init__(self, papers: List[Dict]):
        self._papers = papers
        self._embeddings = None
        self._ready = False
        self._model = None

        if not papers:
            return

        self._model = _get_model()
        if self._model is None:
            # BM25 fallback — still token-overlap but Okapi-B25 weighted
            self._build_bm25(papers)
            return

        try:
            texts = [
                f"{p.get('title', '')}. {p.get('abstract', '')[:500]}"
                for p in papers
            ]
            self._embeddings = self._model.encode(
                texts, convert_to_tensor=True, show_progress_bar=False, batch_size=32
            )
            self._ready = True
        except Exception as e:
            print(f"[CitationManager] Semantic index failed: {e} — using BM25 fallback")
            self._build_bm25(papers)

    def _build_bm25(self, papers: List[Dict]):
        """BM25 fallback when SBERT unavailable."""
        self._bm25_docs = []
        for p in papers:
            text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
            tokens = re.findall(r"[a-z0-9]+", text)
            self._bm25_docs.append(tokens)
        if _BM25_OK and self._bm25_docs:
            self._bm25 = BM25Okapi(self._bm25_docs)
        else:
            self._bm25 = None
        self._bm25_ready = True

    def query(self, sentence: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Returns [(paper_index, similarity_score)] sorted descending.
        Uses PURE SEMANTIC similarity — no keyword matching.
        """
        if self._ready and self._model is not None and self._embeddings is not None:
            try:
                q_emb = self._model.encode(sentence, convert_to_tensor=True)
                scores = st_util.cos_sim(q_emb, self._embeddings)[0].cpu().tolist()
                ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
                return ranked[:top_k]
            except Exception:
                pass

        # BM25 fallback
        if hasattr(self, "_bm25") and self._bm25 is not None:
            tokens = re.findall(r"[a-z0-9]+", sentence.lower())
            scores_arr = self._bm25.get_scores(tokens)
            max_s = max(scores_arr) if max(scores_arr) > 0 else 1.0
            ranked = sorted(enumerate(scores_arr / max_s), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]

        # Last resort: uniform-zero
        return [(i, 0.0) for i in range(min(top_k, len(self._papers)))]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CITATION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedCitationManager:

    def __init__(self, citation_format: CitationFormat = CitationFormat.IEEE):
        self.format        = citation_format
        self.citations     = []        # CSL-JSON dicts
        self.citation_map  = {}        # paper_id → citation_number
        self._raw_papers: List[Dict] = []
        self._sem_index: Optional[_SemanticPaperIndex] = None
        self.bib_style = _load_csl_style(citation_format.value)

    # ── Paper management ───────────────────────────────────────────────────────

    def add_paper(self, paper: Dict) -> int:
        citation_id = len(self.citations) + 1
        if paper.get("doi"):
            paper = self._enrich_from_crossref(paper)
        csl_item = self._to_csl_json(paper, citation_id)
        self.citations.append(csl_item)
        self._raw_papers.append(paper)
        self.citation_map[paper.get("id", str(citation_id))] = citation_id
        self._sem_index = None  # invalidate index
        return citation_id

    def _get_index(self) -> _SemanticPaperIndex:
        if self._sem_index is None:
            self._sem_index = _SemanticPaperIndex(self._raw_papers)
        return self._sem_index

    # ── Crossref enrichment ────────────────────────────────────────────────────

    def _enrich_from_crossref(self, paper: Dict) -> Dict:
        doi = paper.get("doi", "")
        if not doi:
            return paper
        try:
            r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=5)
            if r.status_code == 200:
                data = r.json()["message"]
                paper["journal"]   = data.get("container-title", [""])[0]
                paper["volume"]    = data.get("volume", "")
                paper["issue"]     = data.get("issue", "")
                paper["pages"]     = data.get("page", "")
                paper["publisher"] = data.get("publisher", "")
                if "published-print" in data:
                    dp = data["published-print"]["date-parts"][0]
                    paper["year"] = dp[0] if dp else paper.get("year")
        except Exception:
            pass
        return paper

    # ── CSL conversion ─────────────────────────────────────────────────────────

    def _to_csl_json(self, paper: Dict, citation_id: int) -> Dict:
        authors = self._parse_authors(paper.get("authors", ""))
        csl = {
            "id":              str(citation_id),
            "type":            "article-journal",
            "title":           paper.get("title", ""),
            "author":          authors,
            "issued":          {"date-parts": [[paper.get("year", datetime.now().year)]]},
            "abstract":        paper.get("abstract", ""),
            "DOI":             paper.get("doi", ""),
            "URL":             f"https://arxiv.org/abs/{paper.get('id', '')}",
            "container-title": paper.get("journal", "arXiv preprint"),
            "volume":          paper.get("volume", ""),
            "issue":           paper.get("issue", ""),
            "page":            paper.get("pages", ""),
            "publisher":       paper.get("publisher", ""),
        }
        return {k: v for k, v in csl.items() if v}

    def _parse_authors(self, authors_string: str) -> List[Dict]:
        if not authors_string:
            return [{"family": "Unknown", "given": ""}]
        authors = []
        for author in re.split(r',\s*|\s+and\s+', str(authors_string))[:20]:
            author = author.strip()
            if not author:
                continue
            if ',' in author:
                parts  = author.split(',', 1)
                family = parts[0].strip()
                given  = parts[1].strip()
            else:
                parts  = author.split()
                family = parts[-1] if parts else author
                given  = ' '.join(parts[:-1]) if len(parts) > 1 else ''
            authors.append({"family": family, "given": given})
        return authors or [{"family": "Unknown", "given": ""}]

    # ═══════════════════════════════════════════════════════════════════════════
    # SEMANTIC CITATION INJECTION  (pure embedding similarity, no keywords)
    # ═══════════════════════════════════════════════════════════════════════════

    def insert_citations_smart(
        self,
        text: str,
        section_type: str = "default",
    ) -> Tuple[str, List[Dict]]:
        """
        Insert IEEE-style inline citations based PURELY on semantic similarity.
        No regex triggers, no keyword lists, no hardcoded domain terms.

        Algorithm:
          1. Split text into sentences.
          2. Encode all sentences in one batch (efficient).
          3. For each sentence, compute cosine_sim against all paper embeddings.
          4. If max_sim >= _SEM_THRESHOLD and we still need citations → inject.
          5. Each paper cited at most once per section (variety).
          6. Respect density_target per section.
        """
        if not self.citations or not text.strip():
            return text, []

        density_target = _DENSITY_TARGETS.get(section_type, _DENSITY_TARGETS["default"])
        if density_target == 0:
            return text, []

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        total = len(sentences)
        target_count = max(1, int(total * density_target))

        index = self._get_index()
        model = _get_model()

        # Batch-encode all sentences for efficiency
        sent_embeddings = None
        if model is not None and index._ready:
            try:
                sent_embeddings = model.encode(
                    sentences, convert_to_tensor=True,
                    show_progress_bar=False, batch_size=64
                )
            except Exception:
                sent_embeddings = None

        modified_sentences = list(sentences)
        report: List[Dict] = []
        injected = 0
        cited_papers: set = set()  # avoid citing same paper twice per section

        for i, sent in enumerate(sentences):
            if injected >= target_count:
                break
            if len(sent.split()) < 6:
                continue  # too short to need a citation

            # Get top matching papers for this sentence
            if sent_embeddings is not None:
                try:
                    sent_emb = sent_embeddings[i]
                    scores = st_util.cos_sim(sent_emb, index._embeddings)[0].cpu().tolist()
                    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
                except Exception:
                    ranked = index.query(sent, top_k=3)
            else:
                ranked = index.query(sent, top_k=3)

            # Find best uncited paper above threshold
            best_idx, best_score, cit_num = None, 0.0, None
            for paper_idx, score in ranked[:5]:
                if score < _SEM_THRESHOLD:
                    break
                num = paper_idx + 1
                if num not in cited_papers:
                    best_idx, best_score, cit_num = paper_idx, score, num
                    break

            if cit_num is None or best_score < _SEM_THRESHOLD:
                # Still log for grounding report
                if ranked:
                    top_score = ranked[0][1]
                    report.append({
                        "sentence":     sent[:80],
                        "citation_num": None,
                        "score":        round(float(top_score), 3),
                        "supported":    False,
                    })
                continue

            # Inject citation
            marker = self._format_inline_citation(cit_num)
            if sent and sent[-1] in '.!?':
                modified_sentences[i] = sent[:-1] + f" {marker}" + sent[-1]
            else:
                modified_sentences[i] = sent + f" {marker}"

            cited_papers.add(cit_num)
            injected += 1
            report.append({
                "sentence":     sent[:80],
                "citation_num": cit_num,
                "score":        round(float(best_score), 3),
                "supported":    best_score >= _SUPPORT_THRESHOLD,
            })

        return " ".join(modified_sentences), report

    def get_unsupported_sentences(self, report: List[Dict]) -> List[str]:
        return [
            r["sentence"] for r in report
            if r.get("supported") is False and r.get("citation_num") is None
        ]

    # ── Inline citation formatting ─────────────────────────────────────────────

    def _format_inline_citation(self, citation_num: int) -> str:
        if self.format == CitationFormat.IEEE:
            return f"[{citation_num}]"
        if citation_num <= 0 or citation_num > len(self.citations):
            return f"[{citation_num}]"
        paper  = self.citations[citation_num - 1]
        author = paper.get("author", [{}])[0].get("family", "Unknown")
        year   = paper.get("issued", {}).get("date-parts", [[""]])[0][0]
        fmt = self.format
        if fmt == CitationFormat.APA:
            return f"({author}, {year})"
        if fmt == CitationFormat.MLA:
            return f"({author})"
        if fmt in (CitationFormat.CHICAGO, CitationFormat.HARVARD):
            return f"({author} {year})"
        if fmt == CitationFormat.VANCOUVER:
            return f"({citation_num})"
        return f"[{citation_num}]"

    # ── Bibliography formatting ────────────────────────────────────────────────

    def format_bibliography(self) -> List[str]:
        if not self.citations:
            return []
        if not _CITEPROC_OK or self.bib_style is None:
            return self._format_bibliography_fallback()
        try:
            bib_source   = CiteProcJSON(self.citations)
            bibliography = CitationStylesBibliography(
                self.bib_style, bib_source, formatter=_plain_formatter,
            )
            for cit in self.citations:
                bibliography.register(Citation([CitationItem(cit["id"])]))
            refs = []
            for item in bibliography.bibliography():
                ref_text = re.sub(r'<[^>]+>', '', str(item))
                refs.append(ref_text)
            return refs
        except Exception as e:
            print(f"[CitationManager] citeproc failed: {e} — using fallback")
            return self._format_bibliography_fallback()

    def _format_bibliography_fallback(self) -> List[str]:
        refs = []
        for i, cit in enumerate(self.citations, 1):
            authors = ", ".join(
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in cit.get("author", [])[:3]
            )
            if not authors:
                authors = "Unknown"
            year  = cit.get("issued", {}).get("date-parts", [[""]])[0][0]
            title = cit.get("title", "Untitled")
            venue = cit.get("container-title", "arXiv preprint")
            doi   = cit.get("DOI", "")
            url   = cit.get("URL", "")
            ref   = f'[{i}] {authors}, "{title}," {venue}, {year}.'
            if doi:
                ref += f" doi: {doi}."
            elif url:
                ref += f" [Online]. Available: {url}"
            refs.append(ref)
        return refs

    def export_bibtex(self) -> str:
        entries = []
        for cit in self.citations:
            entry_id = f"ref{cit['id']}"
            entry    = f"@article{{{entry_id},\n"
            entry   += f"  title   = {{{cit.get('title', '')}}},\n"
            if cit.get("author"):
                authors = " and ".join(
                    f"{a.get('given', '')} {a.get('family', '')}"
                    for a in cit["author"]
                )
                entry += f"  author  = {{{authors}}},\n"
            if cit.get("issued"):
                entry += f"  year    = {{{cit['issued']['date-parts'][0][0]}}},\n"
            if cit.get("container-title"):
                entry += f"  journal = {{{cit['container-title']}}},\n"
            if cit.get("DOI"):
                entry += f"  doi     = {{{cit['DOI']}}},\n"
            if cit.get("URL"):
                entry += f"  url     = {{{cit['URL']}}},\n"
            entry += "}\n"
            entries.append(entry)
        return "\n".join(entries)

    def get_stats(self) -> Dict:
        if not self.citations:
            return {}
        years = [
            c["issued"]["date-parts"][0][0]
            for c in self.citations
            if c.get("issued")
        ]
        return {
            "total_citations": len(self.citations),
            "format":          self.format.value,
            "year_range":      {"min": min(years) if years else None,
                                "max": max(years) if years else None},
            "with_doi":        sum(1 for c in self.citations if c.get("DOI")),
            "unique_authors":  len({
                a["family"]
                for c in self.citations
                for a in c.get("author", [])
            }),
            "semantic_index_ready": (
                self._sem_index is not None and self._sem_index._ready
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PAPER EVALUATION METRICS  (Perplexity, BLEU, ROUGE)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_paper(generated_text: str, reference_texts: List[str] = None) -> Dict:
    """
    Compute NLP quality metrics for the generated paper.

    Metrics:
      - Perplexity  : log-likelihood proxy (bigram language model). Lower = better.
      - BLEU        : n-gram precision vs references (or self-bigram). 0-1 scale.
      - ROUGE-1/2/L : recall/precision/F1 overlap vs references. 0-1 scale.
      - Avg sentence length, lexical diversity (type-token ratio).

    If reference_texts is empty, uses the generated text itself for ROUGE self-eval
    (shows structural diversity). BLEU is then computed as a unigram self-score.
    """
    if not generated_text or len(generated_text.strip()) < 50:
        return {"error": "Text too short to evaluate"}

    tokens = re.findall(r"[a-zA-Z0-9']+", generated_text.lower())
    if len(tokens) < 10:
        return {"error": "Insufficient tokens"}

    # ── Perplexity (bigram log-prob estimate) ─────────────────────────────────
    perplexity = _compute_perplexity(tokens)

    # ── BLEU ─────────────────────────────────────────────────────────────────
    if reference_texts:
        bleu = _compute_bleu(generated_text, reference_texts)
    else:
        bleu = _compute_self_bleu(generated_text)  # self-diversity score

    # ── ROUGE ─────────────────────────────────────────────────────────────────
    if reference_texts:
        rouge = _compute_rouge(generated_text, reference_texts)
    else:
        rouge = _compute_rouge_self(generated_text)

    # ── Lexical diversity ──────────────────────────────────────────────────────
    sentences = re.split(r'[.!?]+', generated_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_sent_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    ttr = len(set(tokens)) / max(len(tokens), 1)  # type-token ratio

    return {
        "perplexity":          round(perplexity, 2),
        "bleu_score":          round(bleu, 4),
        "rouge_1_f1":          round(rouge["rouge1_f"], 4),
        "rouge_2_f1":          round(rouge["rouge2_f"], 4),
        "rouge_l_f1":          round(rouge["rougeL_f"], 4),
        "avg_sentence_length": round(avg_sent_len, 1),
        "lexical_diversity":   round(ttr, 4),
        "total_words":         len(tokens),
        "unique_words":        len(set(tokens)),
        "interpretation": {
            "perplexity":   "Lower is better (< 50 = coherent, 50-200 = acceptable, > 200 = incoherent)",
            "bleu":         "Higher is better (0.3+ good vs reference, 0.5+ excellent)",
            "rouge":        "Higher recall/precision vs reference; self-eval shows internal diversity",
            "diversity":    "Type-token ratio: > 0.5 = varied vocabulary",
        }
    }


def _compute_perplexity(tokens: List[str]) -> float:
    """Bigram language model perplexity."""
    if len(tokens) < 2:
        return 999.9

    unigrams = Counter(tokens)
    bigrams  = Counter(zip(tokens[:-1], tokens[1:]))
    N = len(tokens)

    log_prob = 0.0
    vocab_size = len(unigrams)
    for i in range(1, N):
        bigram  = (tokens[i-1], tokens[i])
        bg_cnt  = bigrams.get(bigram, 0)
        ug_cnt  = unigrams.get(tokens[i-1], 0)
        # Laplace smoothing
        prob = (bg_cnt + 1) / (ug_cnt + vocab_size)
        log_prob += math.log(prob)

    avg_log_prob = log_prob / (N - 1)
    return math.exp(-avg_log_prob)


def _compute_bleu(hypothesis: str, references: List[str], max_n: int = 4) -> float:
    """BLEU score with brevity penalty."""
    hyp_tokens = re.findall(r"[a-zA-Z0-9']+", hypothesis.lower())
    if not hyp_tokens:
        return 0.0

    bp = 1.0
    ref_len = min(len(re.findall(r"[a-zA-Z0-9']+", r.lower())) for r in references)
    if len(hyp_tokens) < ref_len:
        bp = math.exp(1 - ref_len / max(len(hyp_tokens), 1))

    precision_scores = []
    for n in range(1, min(max_n + 1, len(hyp_tokens))):
        hyp_ngrams = Counter(_ngrams(hyp_tokens, n))
        max_ref_count = Counter()
        for ref in references:
            ref_tokens = re.findall(r"[a-zA-Z0-9']+", ref.lower())
            ref_ngrams = Counter(_ngrams(ref_tokens, n))
            for ngram in hyp_ngrams:
                max_ref_count[ngram] = max(max_ref_count[ngram], ref_ngrams[ngram])
        clipped = sum(min(cnt, max_ref_count[ng]) for ng, cnt in hyp_ngrams.items())
        total   = sum(hyp_ngrams.values())
        precision_scores.append(clipped / max(total, 1))

    if not precision_scores or all(p == 0 for p in precision_scores):
        return 0.0

    log_avg = sum(math.log(p + 1e-10) for p in precision_scores) / len(precision_scores)
    return bp * math.exp(log_avg)


def _compute_self_bleu(text: str) -> float:
    """Self-BLEU: measures sentence diversity (lower self-BLEU = more diverse)."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 4]
    if len(sentences) < 2:
        return 0.0
    total = 0.0
    for i, sent in enumerate(sentences[:20]):
        refs = [sentences[j] for j in range(len(sentences)) if j != i][:5]
        total += _compute_bleu(sent, refs, max_n=2)
    return total / min(20, len(sentences))


def _compute_rouge(hypothesis: str, references: List[str]) -> Dict:
    """ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    hyp_tokens = re.findall(r"[a-zA-Z0-9']+", hypothesis.lower())

    def rouge_n(hyp, ref_tokens, n):
        hyp_ng = Counter(_ngrams(hyp, n))
        ref_ng = Counter(_ngrams(ref_tokens, n))
        overlap = sum(min(hyp_ng[ng], ref_ng[ng]) for ng in hyp_ng)
        precision = overlap / max(sum(hyp_ng.values()), 1)
        recall    = overlap / max(sum(ref_ng.values()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        return f1

    def lcs_length(a, b):
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(2)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i-1] == b[j-1]:
                    dp[i%2][j] = dp[(i-1)%2][j-1] + 1
                else:
                    dp[i%2][j] = max(dp[(i-1)%2][j], dp[i%2][j-1])
        return dp[m%2][n]

    r1_scores, r2_scores, rl_scores = [], [], []
    for ref in references:
        ref_tokens = re.findall(r"[a-zA-Z0-9']+", ref.lower())
        r1_scores.append(rouge_n(hyp_tokens, ref_tokens, 1))
        r2_scores.append(rouge_n(hyp_tokens, ref_tokens, 2))
        lcs = lcs_length(hyp_tokens[:200], ref_tokens[:200])
        p = lcs / max(len(hyp_tokens), 1)
        r = lcs / max(len(ref_tokens), 1)
        rl_scores.append(2*p*r / max(p+r, 1e-9))

    return {
        "rouge1_f": sum(r1_scores) / max(len(r1_scores), 1),
        "rouge2_f": sum(r2_scores) / max(len(r2_scores), 1),
        "rougeL_f": sum(rl_scores) / max(len(rl_scores), 1),
    }


def _compute_rouge_self(text: str) -> Dict:
    """ROUGE on sentence pairs within the document (structural diversity)."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 4]
    if len(sentences) < 2:
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0}
    sample = sentences[:10]
    half = len(sample) // 2
    hyp_block = " ".join(sample[:half])
    ref_block  = " ".join(sample[half:])
    return _compute_rouge(hyp_block, [ref_block])


def _ngrams(tokens: List[str], n: int) -> List[Tuple]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]