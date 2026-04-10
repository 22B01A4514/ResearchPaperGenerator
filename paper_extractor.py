"""
paper_extractor.py  —  v2
==========================
Fixes in v2:
  1. Chunking for RAG — each paper split into 300-500 token overlapping chunks
     with chunk_index stored so retrieval is precise, not just "whole paper"
  2. Reference section stripping — removes [1]...[N] blocks before LLM context
  3. IEEE-aware title extraction — skips author lines, affiliation lines
  4. Extraction confidence score — based on abstract found, title quality, length
  5. Duplicate detection — md5 hash of first 2000 chars, deduplicates across calls
  6. Language detection hint — warns if paper appears non-English

Supported formats: PDF (pdfplumber → PyPDF2 fallback), DOCX
"""

import re
import os
import hashlib
from typing import Dict, Optional, List, Tuple
from pathlib import Path

# ── Duplicate tracking (module-level, persists for session) ──────────────────
_SEEN_HASHES: set = set()


# ── PDF extraction ────────────────────────────────────────────────────────────

def _extract_pdf(filepath: str) -> str:
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(filepath) as pdf:
            pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
            text = "\n".join(pages)
        if text.strip():
            return text
    except Exception:
        pass
    try:
        import PyPDF2
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages = [p.extract_text() for p in reader.pages if p.extract_text()]
            text = "\n".join(pages)
    except Exception as e:
        print(f"[paper_extractor] PDF extraction failed for {filepath}: {e}")
    return text


def _extract_docx(filepath: str) -> str:
    try:
        from docx import Document
        doc = Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        print(f"[paper_extractor] DOCX extraction failed for {filepath}: {e}")
        return ""


# ── Reference section stripping ───────────────────────────────────────────────

def _strip_references(text: str) -> str:
    """
    Remove the References / Bibliography section from extracted text.
    This prevents [1] Author, Title... lines from polluting LLM prompts.
    """
    # Match common reference section headers
    pattern = re.compile(
        r'\n\s*(?:References|Bibliography|Works Cited|REFERENCES|BIBLIOGRAPHY)\s*\n.*',
        re.DOTALL | re.IGNORECASE,
    )
    stripped = pattern.sub("", text)
    # Also strip inline [N] citation lines at end of paragraphs
    stripped = re.sub(r'\[\d+\]\s+[A-Z][^.\n]{10,120}\.\s*\n', '', stripped)
    return stripped.strip()


# ── Metadata heuristics ────────────────────────────────────────────────────────

def _guess_title(text: str, filename: str) -> Tuple[str, float]:
    """
    Returns (title, confidence).
    IEEE papers: title is usually lines 1-3, before author block.
    Confidence: 0.0-1.0
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Skip known non-title patterns
    skip_patterns = [
        r'^\d{4}',                          # starts with year
        r'^(Abstract|Keywords?|Index Terms)',# section headers
        r'@',                               # email
        r'^\d+$',                           # page numbers
        r'University|Institute|Department|IEEE|ACM|arXiv',  # affiliations
        r'^[A-Z][a-z]+,?\s+[A-Z]\.',       # "Lastname, F." author format
    ]

    for line in lines[:20]:
        words = line.split()
        if len(words) < 3 or len(words) > 25:
            continue
        if any(re.search(p, line) for p in skip_patterns):
            continue
        if not line[0].isupper():
            continue
        if line.endswith('.') and len(words) < 5:
            continue
        # Looks like a title
        confidence = min(1.0, 0.5 + 0.05 * len(words))
        return line, confidence

    stem = Path(filename).stem
    return stem.replace("_", " ").replace("-", " ").title(), 0.3


def _guess_authors(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    author_pattern = re.compile(
        r'^([A-Z][a-z][\w\-]+ (?:[A-Z]\.? )?[A-Z][a-z][\w\-]+'
        r'(?:,\s*[A-Z][a-z][\w\-]+ (?:[A-Z]\.? )?[A-Z][a-z][\w\-]+){0,8})$'
    )
    for line in lines[1:25]:
        if author_pattern.match(line):
            return line
    return "Uploaded Paper"


def _guess_year(text: str) -> Optional[int]:
    years = re.findall(r'\b(19[9][0-9]|20[0-2][0-9])\b', text[:3000])
    if not years:
        return None
    from collections import Counter
    return int(Counter(years).most_common(1)[0][0])


def _extract_abstract(text: str) -> str:
    match = re.search(
        r'\bAbstract\b[\s\-:—]*\n?(.*?)(?=\n\s*\n|\b1[\.\s]+Introduction|\bKeywords?\b|\bIndex Terms\b)',
        text, re.IGNORECASE | re.DOTALL,
    )
    if match:
        abstract = re.sub(r'\s+', ' ', match.group(1)).strip()
        if len(abstract) > 100:
            return abstract[:1200]
    clean = re.sub(r'\s+', ' ', text[:2000]).strip()
    return clean[:800]


def _compute_hash(text: str) -> str:
    return hashlib.md5(text[:2000].encode("utf-8", errors="replace")).hexdigest()


def _detect_language(text: str) -> str:
    """Very lightweight: count English stopwords vs total words."""
    sample = text[:500].lower().split()
    en_stops = {"the", "of", "and", "in", "a", "to", "is", "for", "this", "that",
                "with", "we", "are", "on", "an", "by", "as", "from", "be", "or"}
    hits = sum(1 for w in sample if w in en_stops)
    ratio = hits / max(len(sample), 1)
    return "en" if ratio > 0.05 else "unknown"


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, max_tokens: int = 400, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks of ~max_tokens words each.
    Simple word-count proxy for tokens (1 token ≈ 0.75 words).
    """
    words = text.split()
    chunk_size = int(max_tokens * 0.75)   # words per chunk
    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks if chunks else [text[:2000]]


# ── Confidence scoring ────────────────────────────────────────────────────────

def _extraction_confidence(
    text: str,
    abstract: str,
    title_conf: float,
) -> float:
    """Score 0-1 based on extraction quality signals."""
    score = 0.0
    score += 0.3 if len(text) > 1000 else len(text) / 1000 * 0.3
    score += 0.3 if len(abstract) > 150 else len(abstract) / 150 * 0.3
    score += 0.2 * title_conf
    score += 0.2 if _guess_year(text) is not None else 0.0
    return round(min(1.0, score), 2)


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_paper(filepath: str, deduplicate: bool = True) -> Optional[Dict]:
    """
    Extract a single uploaded file into a paper dict with chunks.
    Returns None on failure or duplicate.
    """
    filepath = str(filepath)
    filename = os.path.basename(filepath)
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        raw_text = _extract_pdf(filepath)
    elif ext in (".docx", ".doc"):
        raw_text = _extract_docx(filepath)
    else:
        print(f"[paper_extractor] Unsupported format: {ext}")
        return None

    if not raw_text or len(raw_text.strip()) < 100:
        print(f"[paper_extractor] Could not extract meaningful text from {filename}")
        return None

    # ── Duplicate detection ───────────────────────────────────────────────────
    doc_hash = _compute_hash(raw_text)
    if deduplicate and doc_hash in _SEEN_HASHES:
        print(f"[paper_extractor] Duplicate detected, skipping: {filename}")
        return None
    if deduplicate:
        _SEEN_HASHES.add(doc_hash)

    # ── Strip references before metadata extraction ───────────────────────────
    clean_text = _strip_references(raw_text)

    # ── Language check ────────────────────────────────────────────────────────
    lang = _detect_language(clean_text)
    if lang != "en":
        print(f"[paper_extractor] Warning: {filename} may not be English (lang={lang})")

    # ── Metadata ──────────────────────────────────────────────────────────────
    title, title_conf = _guess_title(clean_text, filename)
    authors   = _guess_authors(clean_text)
    year      = _guess_year(clean_text)
    abstract  = _extract_abstract(clean_text)
    confidence = _extraction_confidence(clean_text, abstract, title_conf)

    # ── Chunking for RAG ──────────────────────────────────────────────────────
    # Use body text (after abstract) for chunking
    body_start = clean_text.find(abstract[:50]) + len(abstract) if abstract else 0
    body_text  = clean_text[body_start:] if body_start > 0 else clean_text
    chunks     = _chunk_text(body_text, max_tokens=400, overlap=50)

    stem     = Path(filename).stem
    paper_id = re.sub(r'[^a-z0-9_]', '_', stem.lower())

    # Full text for LLM context (truncated, references stripped)
    full_text_truncated = re.sub(r'\s+', ' ', clean_text).strip()[:4000]

    base_paper = {
        "id":                   f"upload_{paper_id}",
        "title":                title,
        "authors":              authors,
        "year":                 year,
        "abstract":             abstract,
        "full_text":            full_text_truncated,
        "source":               "upload",
        "filename":             filename,
        "doi":                  "",
        "journal_ref":          "User-Uploaded Paper",
        "categories":           "",
        "relevance_score":      1.0,
        "extraction_confidence": confidence,
        "language":             lang,
        "doc_hash":             doc_hash,
        # Chunks for fine-grained RAG retrieval
        "chunks": [
            {
                "chunk_index": i,
                "text":        chunk,
                "paper_id":    f"upload_{paper_id}",
            }
            for i, chunk in enumerate(chunks)
        ],
    }

    return base_paper


def extract_papers(filepaths: List[str], deduplicate: bool = True) -> List[Dict]:
    """Extract multiple uploaded papers. Skips files that fail or are duplicates."""
    results = []
    for fp in filepaths:
        paper = extract_paper(fp, deduplicate=deduplicate)
        if paper:
            results.append(paper)
            print(f"[paper_extractor] ✓ Extracted: {paper['title'][:60]}... "
                  f"(confidence={paper['extraction_confidence']}, "
                  f"chunks={len(paper['chunks'])})")
        else:
            print(f"[paper_extractor] ✗ Skipped: {fp}")
    return results


def clear_duplicate_cache():
    """Call between sessions if needed."""
    global _SEEN_HASHES
    _SEEN_HASHES = set()