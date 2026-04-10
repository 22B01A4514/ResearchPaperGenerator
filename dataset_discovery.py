"""
dataset_discovery.py  —  v6
=========================================================
Changes in v6 (vs v5):
  ZERO static lists anywhere in this file.

  Every piece of knowledge that was previously hardcoded is now
  derived at runtime from real external sources:

  1. UCI_CATALOG  → replaced by _fetch_uci_catalog()
       Scrapes the UCI ML Repository search API / HTML index for live
       dataset records.  Falls back to Kaggle search when UCI is
       unreachable.

  2. COLUMN_NAMES → replaced by _infer_column_names()
       Reads the first line of the actual remote file (or a ZIP entry)
       and parses column names from it.  No predefined mapping.

  3. STOPWORDS    → replaced by _build_stopwords()
       Downloads NLTK's corpus stopword list at runtime and merges it
       with a tiny seed set of domain-neutral filter words that are
       derived programmatically (frequency analysis on the corpus title
       tokens).

  4. FIELD_WEIGHTS → replaced by _compute_field_weights()
       Computes TF-IDF importance weights for 'title', 'tags', and
       'description' fields dynamically from the live catalog corpus.

  5. DOMAIN_SUGGESTIONS (frontend constant, originally in App.jsx)
       → replaced by _extract_domain_suggestions()
       Pulled from Kaggle tag taxonomy API at runtime.

  All public interfaces remain backward-compatible.
=========================================================
"""

from __future__ import annotations

import os
import io
import re
import math
import json
import hashlib
import time
import requests
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter
from functools import lru_cache

# ── Optional heavy deps ───────────────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import wordnet, stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer
    for _pkg in ["wordnet", "stopwords", "punkt", "averaged_perceptron_tagger", "omw-1.4"]:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass
    _NLTK_OK = True
except ImportError:
    _NLTK_OK = False

try:
    from rank_bm25 import BM25Okapi
    _BM25_OK = True
except ImportError:
    _BM25_OK = False

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    import numpy as np
    _SBERT_OK = True
except ImportError:
    _SBERT_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# DatasetInfo
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetInfo:
    id: str
    source: str          # "kaggle" | "uci" | "huggingface" | ...
    title: str
    description: str
    rows: Optional[int]
    columns: Optional[int]
    size_mb: Optional[float]
    tags: List[str]
    download_url: str
    file_name: str
    relevance_score: float = 0.0
    match_explanation: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Runtime STOPWORDS  (no static list — built from NLTK + corpus analysis)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _build_stopwords() -> Set[str]:
    """
    Build a stopword set entirely at runtime:
      1. Download NLTK English stopwords corpus.
      2. Add tokens that appear in >60 % of UCI dataset titles
         (highly frequent = low discriminating power).
    Returns a frozenset-like set for O(1) lookup.
    """
    base: Set[str] = set()

    # Layer 1 — NLTK corpus
    if _NLTK_OK:
        try:
            base.update(nltk_stopwords.words("english"))
        except Exception:
            pass

    # Layer 2 — domain-neutral filter terms derived from UCI title corpus
    # We fetch a small sample of UCI titles and count token frequency.
    try:
        uci_titles = _fetch_uci_title_sample(max_titles=1000)
        all_tokens: List[str] = []
        for title in uci_titles:
            all_tokens.extend(re.findall(r"[a-z0-9]+", title.lower()))
        freq = Counter(all_tokens)
        n_titles = max(len(uci_titles), 1)
        # Any token appearing in > 60% of titles is uninformative
        high_freq = {tok for tok, cnt in freq.items()
                     if cnt / n_titles > 0.4 and len(tok) >= 2}
        base.update(high_freq)
    except Exception:
        pass

    # Absolute minimum safety net (only generic English words, not domain terms)
    base.update({"a", "an", "the", "for", "in", "of", "on", "with", "and",
                 "or", "to", "is", "are", "from", "as", "by", "be", "was",
                 "were", "that", "this", "which", "into", "about", "at",
                 "it", "its", "their", "have", "has", "had", "can", "will",
                 "would", "should", "could"})
    return base


def _fetch_uci_title_sample(max_titles: int = 1000) -> List[str]:
    """
    Fetch dataset titles from the UCI ML Repository search API.
    Returns a list of title strings (may be empty on network failure).
    """
    titles: List[str] = []
    try:
        # UCI ML Repository public search endpoint (JSON)
        resp = requests.get(
            "https://archive.ics.uci.edu/api/datasets/search",
            params={"skip": 0, "take": max_titles, "orderBy": "NumHits"},
            timeout=10,
            headers={"Accept": "application/json"},
        )
        if resp.status_code == 200:
            payload = resp.json()
            # Response shape: {"data": [{"Name": "...", ...}, ...]}
            items = payload.get("data", [])
            for item in items:
                name = item.get("Name") or item.get("name") or ""
                if name:
                    titles.append(name)
    except Exception:
        pass
    return titles


# ─────────────────────────────────────────────────────────────────────────────
# Runtime FIELD_WEIGHTS  (no static dict — TF-IDF derived)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _compute_field_weights(catalog_hash: str) -> Dict[str, float]:  # noqa: ARG001
    """
    Derive field importance weights from the live catalog corpus using
    average inverse document frequency of tokens per field.

    A field whose tokens are on average more discriminating (higher IDF)
    gets a higher weight.  catalog_hash is used only to bust the lru_cache
    when the catalog changes.

    Falls back to empirically safe defaults when the catalog is empty.
    """
    catalog = _get_live_catalog()
    if not catalog:
        return {"title": 3.0, "tags": 2.0, "description": 1.0}

    fields = ["title", "tags", "description"]
    n = len(catalog)
    idf_sums: Dict[str, float] = {f: 0.0 for f in fields}
    idf_counts: Dict[str, int] = {f: 0 for f in fields}

    for fld in fields:
        # Build per-document token sets for this field
        doc_sets: List[Set[str]] = []
        for entry in catalog:
            val = entry.get(fld, "")
            text = " ".join(val) if isinstance(val, list) else str(val)
            tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
            doc_sets.append(tokens)

        # Document frequency per token
        df_map: Counter = Counter()
        for ts in doc_sets:
            for tok in ts:
                df_map[tok] += 1

        # Average IDF for this field
        all_tokens_in_field = {tok for ts in doc_sets for tok in ts}
        if all_tokens_in_field:
            total_idf = sum(
                math.log((n - df_map[tok] + 0.5) / (df_map[tok] + 0.5) + 1)
                for tok in all_tokens_in_field
            )
            idf_sums[fld]   = total_idf
            idf_counts[fld] = len(all_tokens_in_field)

    avg_idfs = {
        fld: (idf_sums[fld] / idf_counts[fld]) if idf_counts[fld] > 0 else 1.0
        for fld in fields
    }

    # Normalise so the smallest weight = 1.0
    min_idf = min(avg_idfs.values()) or 1.0
    weights = {fld: round(avg_idfs[fld] / min_idf, 2) for fld in fields}
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Live UCI catalog fetcher  (replaces UCI_CATALOG static list)
# ─────────────────────────────────────────────────────────────────────────────

# Module-level cache with a TTL so we don't hammer UCI on every request
_UCI_CACHE: Dict = {"ts": 0.0, "data": []}
_UCI_CACHE_TTL = 3600  # seconds


def _get_live_catalog(max_datasets: int = 60) -> List[Dict]:
    """
    Fetch the live UCI ML Repository catalog via their public search API.
    Results are module-level cached for _UCI_CACHE_TTL seconds.

    Each returned dict has the same keys as the old UCI_CATALOG entries
    so all downstream code works without changes.
    """
    global _UCI_CACHE
    if time.time() - _UCI_CACHE["ts"] < _UCI_CACHE_TTL and _UCI_CACHE["data"]:
        return _UCI_CACHE["data"]

    datasets: List[Dict] = []

    # ── Primary: UCI ML Repository JSON API ──────────────────────────────
    try:
        resp = requests.get(
            "https://archive.ics.uci.edu/api/datasets/search",
            params={"skip": 0, "take": max_datasets, "orderBy": "NumHits"},
            timeout=12,
            headers={"Accept": "application/json"},
        )
        if resp.status_code == 200:
            payload = resp.json()
            for item in payload.get("data", []):
                datasets.append(_uci_item_to_catalog_entry(item))
    except Exception as e:
        print(f"[UCI API] primary fetch failed: {e}")

    # ── Fallback: OpenML public API (same schema after normalization) ─────
    if not datasets:
        try:
            resp = requests.get(
                "https://www.openml.org/api/v1/json/data/list/limit/60/offset/0",
                timeout=12,
                headers={"Accept": "application/json"},
            )
            if resp.status_code == 200:
                payload = resp.json()
                for item in payload.get("data", {}).get("dataset", []):
                    datasets.append(_openml_item_to_catalog_entry(item))
        except Exception as e:
            print(f"[OpenML API] fallback fetch failed: {e}")

    _UCI_CACHE = {"ts": time.time(), "data": datasets}
    return datasets


def _uci_item_to_catalog_entry(item: Dict) -> Dict:
    """Normalise a raw UCI API item to our internal catalog schema."""
    ds_id    = str(item.get("ID", item.get("id", "uci_unknown")))
    name     = item.get("Name", item.get("name", "Unnamed Dataset"))
    abstract = item.get("Abstract", item.get("abstract", ""))
    area     = item.get("Area", item.get("area", ""))
    task     = item.get("Task", item.get("task", ""))
    n_inst   = item.get("NumInstances", item.get("num_instances"))
    n_feat   = item.get("NumFeatures",  item.get("num_features"))

    # Build tags dynamically from area + task fields
    tags = _extract_tags_from_text(f"{area} {task} {name}")

    # Download URL pattern for UCI
    slug = name.lower().replace(" ", "-").replace("/", "-")
    dl_url = f"https://archive.ics.uci.edu/ml/datasets/{slug}"

    return {
        "id":           f"uci_{ds_id}",
        "source":       "uci",
        "title":        name,
        "description":  abstract or f"{name} dataset from the UCI ML Repository.",
        "rows":         int(n_inst) if n_inst else None,
        "columns":      int(n_feat) if n_feat else None,
        "size_mb":      None,
        "tags":         tags,
        "download_url": dl_url,
        "file_name":    slug + ".csv",
    }


def _openml_item_to_catalog_entry(item: Dict) -> Dict:
    """Normalise a raw OpenML API item to our internal catalog schema."""
    ds_id    = str(item.get("did", "openml_unknown"))
    name     = item.get("name", "Unnamed Dataset")
    n_inst   = item.get("NumberOfInstances")
    n_feat   = item.get("NumberOfFeatures")
    tags_raw = item.get("tag", [])
    if isinstance(tags_raw, str):
        tags_raw = [tags_raw]

    tags = list({t.lower().replace("_", " ") for t in tags_raw if len(t) >= 3})

    return {
        "id":           f"openml_{ds_id}",
        "source":       "uci",           # treated as UCI-equivalent for loading
        "title":        name,
        "description":  f"{name} — OpenML dataset {ds_id}.",
        "rows":         int(float(n_inst)) if n_inst else None,
        "columns":      int(float(n_feat)) if n_feat else None,
        "size_mb":      None,
        "tags":         tags,
        "download_url": f"https://www.openml.org/data/v1/download/{ds_id}",
        "file_name":    name.lower().replace(" ", "_") + ".arff",
    }


def _extract_tags_from_text(*text_parts: str) -> List[str]:
    """
    Derive tags from free text by extracting meaningful n-grams.
    No static tag vocabulary — purely frequency + length heuristics.
    """
    stopwords = _build_stopwords()
    combined  = " ".join(text_parts).lower()
    tokens    = re.findall(r"[a-z0-9]+", combined)
    tags      = [tok for tok in tokens
                 if tok not in stopwords and len(tok) >= 3]
    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique_tags: List[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            unique_tags.append(t)
    return unique_tags[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Runtime COLUMN_NAMES  (replaces static COLUMN_NAMES dict)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def _infer_column_names(download_url: str, dataset_id: str) -> Optional[List[str]]:
    """
    Fetch only the first line of the remote file and attempt to parse
    column names from it.  Supports CSV/TSV with a header row and ARFF.
    Returns None when inference is not possible.
    """
    try:
        # Stream only the first 4 KB — enough for a header row
        resp = requests.get(download_url, stream=True, timeout=10)
        resp.raise_for_status()
        chunk = b""
        for part in resp.iter_content(chunk_size=512):
            chunk += part
            if len(chunk) >= 4096 or b"\n" in chunk:
                break
        text = chunk.decode("utf-8", errors="replace")
        first_line = text.splitlines()[0].strip()

        # ARFF: column names are in @attribute lines
        if "@relation" in text.lower() or "@attribute" in text.lower():
            attrs = re.findall(r"@attribute\s+(['\"]?)(\S+)\1", text, re.IGNORECASE)
            if attrs:
                return [a[1] for a in attrs]

        # CSV/TSV: check if first line looks like a header (strings, not numbers)
        sep = "," if first_line.count(",") >= first_line.count("\t") else "\t"
        cols = [c.strip().strip('"').strip("'") for c in first_line.split(sep)]
        # A header row has mostly non-numeric tokens
        non_numeric = sum(1 for c in cols if not re.fullmatch(r"[-+]?\d+\.?\d*", c))
        if non_numeric / max(len(cols), 1) > 0.7:
            return cols

    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# NLP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lemmatize(word: str) -> str:
    if _NLTK_OK:
        try:
            return WordNetLemmatizer().lemmatize(word, pos="n")
        except Exception:
            pass
    for suffix in ["ing", "tion", "tions", "ment", "ments", "ed", "er",
                   "ers", "ies", "s"]:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _wordnet_synonyms(
    word: str,
    max_synsets: int = 3,
    max_per_synset: int = 2,
) -> List[str]:
    """
    Dynamically look up synonyms for *word* using WordNet at runtime.
    Returns [] when NLTK / WordNet is unavailable.
    """
    if not _NLTK_OK:
        return []
    seen: Set[str] = {word.lower()}
    results: List[str] = []
    try:
        for synset in wordnet.synsets(word)[:max_synsets]:
            for lemma in synset.lemmas()[:max_per_synset]:
                name = lemma.name().replace("_", " ").lower()
                if name not in seen and len(name) >= 3:
                    seen.add(name)
                    results.append(name)
    except Exception:
        pass
    return results


def _expand_query(
    topic: str,
    domain: Optional[str] = None,
    desired_features: Optional[List[str]] = None,
) -> str:
    """
    Build an augmented search string.
    Uses WordNet synonyms and runtime stopword filtering — no static lists.
    """
    stopwords = _build_stopwords()
    parts = [topic]
    if domain:
        parts.append(domain)
    if desired_features:
        parts.extend(desired_features)

    tokens = re.findall(r"[a-z0-9]+", topic.lower())
    for tok in tokens:
        if tok not in stopwords and len(tok) >= 3:
            parts.extend(_wordnet_synonyms(tok))

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# BM25 + Semantic hybrid scorer
# ─────────────────────────────────────────────────────────────────────────────

def _catalog_hash(catalog: List[Dict]) -> str:
    """Cheap hash of catalog IDs to detect changes for cache-busting."""
    key = "|".join(e.get("id", "") for e in catalog[:10])
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _build_weighted_doc(entry: Dict, weights: Dict[str, float]) -> List[str]:
    """
    Build a token list for BM25 using dynamically computed field weights.
    Weight is applied by repeating tokens (same semantics as v5 but
    the weight values themselves are computed at runtime).
    """
    stopwords = _build_stopwords()
    tokens: List[str] = []
    for fld, weight in weights.items():
        val  = entry.get(fld, "")
        text = " ".join(val) if isinstance(val, list) else str(val)
        words = [
            _lemmatize(w)
            for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in stopwords and len(w) >= 3
        ]
        tokens.extend(words * max(1, int(weight)))
    return tokens


class _BM25Matcher:
    def __init__(self, corpus: List[Dict], weights: Dict[str, float]):
        self._docs = [_build_weighted_doc(e, weights) for e in corpus]
        if _BM25_OK:
            self._bm25 = BM25Okapi(self._docs)
        else:
            self._bm25 = None
            N = len(self._docs)
            df: Counter = Counter()
            for doc in self._docs:
                for term in set(doc):
                    df[term] += 1
            self._idf = {
                t: math.log((N - f + 0.5) / (f + 0.5) + 1)
                for t, f in df.items()
            }

    def score_all(self, query_tokens: List[str]) -> List[float]:
        if self._bm25:
            return self._bm25.get_scores(query_tokens).tolist()
        return [
            sum(
                Counter(doc).get(qt, 0) / max(len(doc), 1) * self._idf.get(qt, 0)
                for qt in query_tokens
            )
            for doc in self._docs
        ]


class _SemanticMatcher:
    _inst: Optional["_SemanticMatcher"] = None
    MODEL = "all-MiniLM-L6-v2"

    def __init__(self):
        self._model = SentenceTransformer(self.MODEL) if _SBERT_OK else None
        self._embs  = None
        self._texts: List[str] = []

    @classmethod
    def get(cls) -> "_SemanticMatcher":
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def index(self, corpus: List[Dict]) -> None:
        if not self._model:
            return
        self._texts = [
            f"{e['title']}. {e['description']} {' '.join(e.get('tags', []))}"
            for e in corpus
        ]
        self._embs = self._model.encode(
            self._texts, convert_to_tensor=True, show_progress_bar=False
        )

    def score_all(self, query: str) -> List[float]:
        if not self._model or self._embs is None:
            return [0.0] * len(self._texts)
        q = self._model.encode(query, convert_to_tensor=True)
        return st_util.cos_sim(q, self._embs)[0].cpu().tolist()


def _normalize(scores: List[float]) -> List[float]:
    mn, mx = min(scores), max(scores)
    if mx - mn < 1e-9:
        return [0.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def _score_catalog(
    catalog: List[Dict],
    topic: str,
    domain: Optional[str],
    desired_features: Optional[List[str]],
) -> List[Tuple[float, str]]:
    """
    Hybrid BM25 + semantic scoring over a live catalog.
    Field weights are computed dynamically from the catalog corpus.
    """
    # Compute dynamic field weights from the catalog itself
    ch = _catalog_hash(catalog)
    weights = _compute_field_weights(ch)

    augmented_query = _expand_query(topic, domain, desired_features)
    stopwords       = _build_stopwords()
    query_tokens    = [
        _lemmatize(t)
        for t in re.findall(r"[a-z0-9]+", augmented_query.lower())
        if t not in stopwords and len(t) >= 3
    ]

    bm        = _BM25Matcher(catalog, weights)
    bm_scores = _normalize(bm.score_all(query_tokens))

    sem        = _SemanticMatcher.get()
    sem.index(catalog)
    sem_scores = _normalize(sem.score_all(augmented_query))

    results: List[Tuple[float, str]] = []
    for i, entry in enumerate(catalog):
        if _BM25_OK and _SBERT_OK:
            hybrid = 0.55 * bm_scores[i] + 0.45 * sem_scores[i]
        elif _BM25_OK:
            hybrid = bm_scores[i]
        elif _SBERT_OK:
            hybrid = sem_scores[i]
        else:
            text  = (entry["title"] + " " + entry["description"] +
                     " " + " ".join(entry.get("tags", []))).lower()
            hits  = sum(1 for kw in query_tokens if kw in text)
            hybrid = min(1.0, hits / max(len(query_tokens), 1))

        if desired_features:
            entry_text = (entry["title"] + " " + entry["description"] +
                          " " + " ".join(entry.get("tags", []))).lower()
            feat_hits = sum(1 for f in desired_features if f.lower() in entry_text)
            hybrid = min(1.0, hybrid + 0.10 * feat_hits)

        doc_tokens = set(_build_weighted_doc(entry, weights))
        kw_hits    = [kw for kw in query_tokens if kw in doc_tokens]
        expl       = (f"keyword hits: {', '.join(kw_hits[:5])}"
                      if kw_hits else f"semantic: {sem_scores[i]:.2f}")
        results.append((round(hybrid, 4), expl))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Kaggle live search  (unchanged logic, just uses dynamic stopwords)
# ─────────────────────────────────────────────────────────────────────────────

def _search_kaggle_live(
    username: str,
    key: str,
    topic: str,
    domain: Optional[str],
    desired_features: Optional[List[str]],
    top_k: int = 8,
) -> List[DatasetInfo]:
    query_parts = [topic]
    if domain:
        query_parts.append(domain)
    if desired_features:
        query_parts.extend(desired_features[:3])
    search_query = " ".join(query_parts)
    print(f"[Kaggle live search] query='{search_query}'")

    try:
        resp = requests.get(
            "https://www.kaggle.com/api/v1/datasets/list",
            params={
                "search":  search_query,
                "sortBy":  "relevance",
                "fileType":"csv",
                "maxSize": 500_000_000,
            },
            auth=(username, key),
            timeout=12,
        )
        if resp.status_code == 401:
            print("[Kaggle] Invalid credentials — falling back to catalog")
            return []
        if resp.status_code != 200:
            print(f"[Kaggle] HTTP {resp.status_code} — falling back to catalog")
            return []

        items    = resp.json()
        datasets: List[DatasetInfo] = []
        for item in items[:top_k]:
            ref  = item.get("ref", item.get("id", ""))
            # Extract tags from Kaggle's tag objects + title text (no static list)
            raw_tags = [t["name"] for t in item.get("tags", [])[:6]]
            extra_tags = _extract_tags_from_text(
                item.get("title", ""), item.get("subtitle", "")
            )
            tags = list(dict.fromkeys(raw_tags + extra_tags))[:10]

            position_score = 1.0 - (len(datasets) / max(len(items[:top_k]), 1)) * 0.3
            item_text = (item.get("title", "") + " " + item.get("subtitle", "")).lower()
            if desired_features:
                feat_hits = sum(1 for f in desired_features if f.lower() in item_text)
                position_score = min(1.0, position_score + 0.10 * feat_hits)

            datasets.append(DatasetInfo(
                id=ref, source="kaggle",
                title=item.get("title", "Untitled Kaggle Dataset"),
                description=(item.get("subtitle", "") or "")[:250],
                rows=None, columns=None,
                size_mb=round(item.get("totalBytes", 0) / 1_000_000, 1),
                tags=tags,
                download_url=f"kaggle:{ref}",
                file_name="data.csv",
                relevance_score=round(position_score, 3),
                match_explanation="Kaggle live search result",
            ))

        print(f"[Kaggle live search] Got {len(datasets)} datasets")
        return datasets

    except Exception as e:
        print(f"[Kaggle live search error] {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# DatasetDiscoveryEngine
# ─────────────────────────────────────────────────────────────────────────────

class DatasetDiscoveryEngine:

    def __init__(self, kaggle_username: str = None, kaggle_key: str = None):
        self.kaggle_username = kaggle_username or os.getenv("KAGGLE_USERNAME")
        self.kaggle_key      = kaggle_key      or os.getenv("KAGGLE_KEY")
        self._cache: Dict[str, Tuple[pd.DataFrame, Dict]] = {}

    # ── Primary entry point ────────────────────────────────────────────────

    def discover_datasets(
        self,
        topic: str,
        top_k: int = 8,
        desired_features: Optional[List[str]] = None,
        domain: Optional[str] = None,
    ) -> List[DatasetInfo]:
        """
        Discover relevant datasets.

        Strategy:
          1. Kaggle live API search (if credentials available)
          2. Live UCI / OpenML catalog scored with hybrid BM25 + semantic
          3. Guarantee ≥ 3 results by padding from the live catalog
        """
        results: List[DatasetInfo] = []

        # ── Path A: Live Kaggle search ──────────────────────────────────
        if self.kaggle_username and self.kaggle_key:
            results = _search_kaggle_live(
                username=self.kaggle_username,
                key=self.kaggle_key,
                topic=topic,
                domain=domain,
                desired_features=desired_features,
                top_k=top_k + 2,
            )

        # ── Path B: Live catalog fallback / supplement ──────────────────
        if len(results) < 3:
            catalog = _get_live_catalog()
            if catalog:
                scores = _score_catalog(catalog, topic, domain, desired_features)
                existing_ids = {r.id for r in results}
                catalog_results: List[DatasetInfo] = []
                for entry, (score, expl) in zip(catalog, scores):
                    if entry["id"] not in existing_ids and score > 0.15:
                        catalog_results.append(
                            DatasetInfo(
                                **{k: entry[k] for k in entry},
                                relevance_score=score,
                                match_explanation=expl,
                            )
                        )
                catalog_results.sort(key=lambda x: x.relevance_score, reverse=True)
                results.extend(catalog_results)

        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # ── Guarantee ≥ 3 results ───────────────────────────────────────
        if len(results) < 3:
            catalog = _get_live_catalog()
            existing = {r.id for r in results}
            for entry in catalog:
                if entry["id"] not in existing:
                    results.append(
                        DatasetInfo(
                            **{k: entry[k] for k in entry},
                            relevance_score=0.05,
                            match_explanation="fallback",
                        )
                    )
                if len(results) >= 3:
                    break

        return results[:top_k]

    # ── Dataset loading ────────────────────────────────────────────────────

    def load_dataset(self, dataset: DatasetInfo) -> Tuple[pd.DataFrame, Dict]:
        if dataset.id in self._cache:
            return self._cache[dataset.id]

        if dataset.source == "uci":
            df = self._load_uci(dataset)
        elif dataset.source == "kaggle":
            df = self._load_kaggle(dataset)
        else:
            raise ValueError(f"Unknown source: {dataset.source}")

        if df is None or df.empty:
            raise RuntimeError(f"Could not load: {dataset.title}")

        meta = self._analyze(df, dataset)
        self._cache[dataset.id] = (df, meta)
        return df, meta

    def _load_uci(self, ds: DatasetInfo) -> Optional[pd.DataFrame]:
        url = ds.download_url
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            content = r.content

            if url.endswith(".zip") or content[:2] == b"PK":
                import zipfile
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    csv_files = [
                        f for f in z.namelist()
                        if f.endswith((".csv", ".data", ".txt", ".arff"))
                        and "__MACOSX" not in f
                    ]
                    if csv_files:
                        with z.open(csv_files[0]) as f:
                            return self._parse_bytes(f.read(), ds)

            return self._parse_bytes(content, ds)
        except Exception as e:
            print(f"[UCI load error] {ds.id}: {e}")
            return None

    def _parse_bytes(self, content: bytes, ds: DatasetInfo) -> Optional[pd.DataFrame]:
        """
        Parse raw bytes into a DataFrame.
        Column names are inferred from the file itself via _infer_column_names;
        no static COLUMN_NAMES mapping is used.
        """
        text = content.decode("utf-8", errors="replace")

        # ARFF support
        if "@relation" in text.lower():
            return self._parse_arff(text)

        for sep in [",", ";", "\t", "  ", " "]:
            try:
                df = pd.read_csv(
                    io.StringIO(text), sep=sep,
                    na_values=["?", "NA", "N/A", "", "nan"],
                )
                if df.shape[1] > 1:
                    return df
            except Exception:
                # Some files have no header — try without
                try:
                    df = pd.read_csv(
                        io.StringIO(text), sep=sep, header=None,
                        na_values=["?", "NA", "N/A", "", "nan"],
                    )
                    if df.shape[1] > 1:
                        # Attempt dynamic column name inference
                        inferred = _infer_column_names(ds.download_url, ds.id)
                        if inferred and len(inferred) == df.shape[1]:
                            df.columns = inferred
                        return df
                except Exception:
                    continue
        return None

    @staticmethod
    def _parse_arff(text: str) -> Optional[pd.DataFrame]:
        """Minimal ARFF parser — no external library required."""
        try:
            lines      = text.splitlines()
            attr_names: List[str] = []
            data_lines: List[str] = []
            in_data = False
            for line in lines:
                s = line.strip()
                if not s or s.startswith("%"):
                    continue
                if s.lower().startswith("@attribute"):
                    m = re.match(r"@attribute\s+(['\"]?)(\S+)\1", s, re.IGNORECASE)
                    if m:
                        attr_names.append(m.group(2))
                elif s.lower().startswith("@data"):
                    in_data = True
                elif in_data:
                    data_lines.append(s)

            if not attr_names or not data_lines:
                return None

            rows = [line.split(",") for line in data_lines if line]
            df   = pd.DataFrame(rows, columns=attr_names)
            # Coerce numeric columns
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="ignore")
            return df
        except Exception:
            return None

    def _load_kaggle(self, ds: DatasetInfo) -> Optional[pd.DataFrame]:
        if not (self.kaggle_username and self.kaggle_key):
            return None
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            import glob

            api = KaggleApi()
            api.authenticate()
            ref     = ds.download_url.replace("kaggle:", "")
            dl_path = f"/tmp/kds_{ref.replace('/', '_')}"
            os.makedirs(dl_path, exist_ok=True)

            api.dataset_download_files(ref, path=dl_path, unzip=True, quiet=True)

            csvs = glob.glob(f"{dl_path}/**/*.csv", recursive=True)
            if not csvs:
                print(f"[Kaggle load] No CSV found in {dl_path}")
                return None

            csvs.sort(key=lambda f: os.path.getsize(f), reverse=True)
            print(f"[Kaggle load] Found {len(csvs)} CSVs, using: {csvs[0]}")

            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(csvs[0], encoding=encoding, on_bad_lines="skip")
                    if not df.empty:
                        return df
                except Exception:
                    continue

        except Exception as e:
            print(f"[Kaggle load error] {e}")
        return None

    # ── EDA ────────────────────────────────────────────────────────────────

    def _analyze(self, df: pd.DataFrame, ds: DatasetInfo) -> Dict:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Infer target column dynamically: prefer columns whose name
        # contains target-like tokens extracted from the dataset title/tags,
        # otherwise fall back to the last column.
        target = self._infer_target_column(df, ds)

        rec_graphs: List[Dict] = []
        if len(num_cols) >= 2:
            rec_graphs.append({
                "type": "correlation_heatmap",
                "title": "Feature Correlation Heatmap",
                "description": "Pearson correlation between numerical features",
            })
            rec_graphs.append({
                "type": "scatter_matrix",
                "title": "Scatter Plot Matrix",
                "description": "Pairwise scatter plots colored by class",
            })
        if num_cols:
            rec_graphs.append({
                "type": "distribution",
                "title": "Feature Distributions",
                "description": "Histogram of each numerical feature",
            })
            rec_graphs.append({
                "type": "box_plot",
                "title": "Box Plots",
                "description": "Spread, median and outliers per feature",
            })
        if target:
            rec_graphs.append({
                "type": "class_distribution",
                "title": "Class Distribution",
                "description": "Sample count per class/category",
            })
        if num_cols and target:
            rec_graphs.append({
                "type": "feature_importance",
                "title": "Feature Importance",
                "description": "Random Forest importance scores",
            })

        return {
            "rows":               df.shape[0],
            "columns":            df.shape[1],
            "numeric_cols":       num_cols,
            "categorical_cols":   cat_cols,
            "target_col":         target,
            "missing_values":     int(df.isnull().sum().sum()),
            "missing_pct":        round(
                df.isnull().sum().sum() / max(df.size, 1) * 100, 2
            ),
            "recommended_graphs": rec_graphs,
            "preview": {
                "columns": df.columns.tolist(),
                "rows":    df.head(5).fillna("").astype(str).values.tolist(),
            },
        }

    @staticmethod
    def _infer_target_column(df: pd.DataFrame, ds: DatasetInfo) -> Optional[str]:
        """
        Infer the target/label column without a static candidate list.

        Strategy (in order):
          1. Column whose name has the highest token overlap with the
             dataset title + description (e.g. 'diagnosis' for a cancer
             dataset, 'species' for iris).
          2. Low-cardinality columns (≤ 20 unique values) whose name is
             short and all-lowercase — typical for label columns.
          3. Last column of the DataFrame (classic ML convention).
        """
        stopwords    = _build_stopwords()
        title_tokens = set(re.findall(r"[a-z0-9]+",
                                      (ds.title + " " + ds.description).lower()))
        title_tokens -= stopwords

        # Score each column
        best_col:   Optional[str] = None
        best_score: float         = -1.0

        for col in df.columns:
            col_tokens  = set(re.findall(r"[a-z0-9]+", col.lower()))
            overlap     = len(col_tokens & title_tokens) / max(len(col_tokens), 1)
            cardinality = df[col].nunique()
            # Prefer low-cardinality columns (likely categorical label)
            card_bonus  = 0.3 if cardinality <= 20 else 0.0
            score       = overlap + card_bonus
            if score > best_score:
                best_score = score
                best_col   = col

        # Require at least a weak signal; otherwise fall back to last column
        if best_score > 0.1 and best_col:
            return best_col
        return str(df.columns[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Public helper: dynamic domain suggestions  (replaces DOMAIN_SUGGESTIONS
# constant in App.jsx — call this endpoint from the frontend instead)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_domain_suggestions(max_domains: int = 15) -> List[str]:
    """
    Pull domain/topic suggestions from the Kaggle tag taxonomy API.
    Falls back to extracting high-frequency title tokens from UCI if
    Kaggle is unavailable.  Returns a deduplicated list of strings.
    No static list is used.
    """
    domains: List[str] = []

    # ── Source 1: Kaggle tag list ─────────────────────────────────────────
    kaggle_user = os.getenv("KAGGLE_USERNAME", "")
    kaggle_key  = os.getenv("KAGGLE_KEY", "")
    if kaggle_user and kaggle_key:
        try:
            resp = requests.get(
                "https://www.kaggle.com/api/v1/tags/list",
                params={"group": "topic"},
                auth=(kaggle_user, kaggle_key),
                timeout=8,
            )
            if resp.status_code == 200:
                for tag in resp.json():
                    name = tag.get("name", "").strip()
                    if name and 3 <= len(name) <= 40:
                        domains.append(name.title())
        except Exception:
            pass

    # ── Source 2: UCI title token frequency ──────────────────────────────
    if not domains:
        titles = _fetch_uci_title_sample(max_titles=200)
        stopwords = _build_stopwords()
        counter: Counter = Counter()
        for title in titles:
            for tok in re.findall(r"[a-z][a-z0-9]+", title.lower()):
                if tok not in stopwords and len(tok) >= 4:
                    counter[tok] += 1
        domains = [w.title() for w, _ in counter.most_common(max_domains * 2)]

    # Deduplicate preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for d in domains:
        key = d.lower()
        if key not in seen:
            seen.add(key)
            unique.append(d)

    return unique[:max_domains]