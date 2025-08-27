# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# Read from .env if set, else use sensible defaults
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "doc_gpt")
CHROMA_DIR = Path(os.getenv("CHROMA_DIR") or (ROOT / "data" / "indexes"))

MIN_PAGE_CHARS_DEFAULT = 400
CHUNK_SIZE_DEFAULT = 800
CHUNK_OVERLAP_DEFAULT = 120


def log(msg: str) -> None:
    print(msg, flush=True)


def sha16(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def load_raw(min_chars: int) -> List[Dict[str, Any]]:
    """Load JSON pages from data/raw; skip corrupt/too-small pages."""
    docs: List[Dict[str, Any]] = []
    paths = sorted(RAW.glob("*.json"))
    if not paths:
        log("No raw docs found. Run scripts/crawl_clean.py first.")
        return docs

    for p in tqdm(paths, desc="Load raw JSON"):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            text = (j.get("text") or "").strip()
            if len(text) < min_chars:
                continue
            docs.append(j)
        except Exception as e:
            log(f"[skip] bad JSON {p.name}: {e!r}")
    return docs


def smart_chunks(text: str, size: int, overlap: int) -> List[str]:
    """
    Greedy chunking that prefers to end near sentence boundaries.
    Works on plain text (already normalized by crawler).
    """
    text = re.sub(r"\s+", " ", text).strip()
    n = len(text)
    if n == 0:
        return []

    chunks: List[str] = []
    i = 0
    while i < n:
        end = min(i + size, n)
        # try to break near a sentence boundary within the last 80 chars of the window
        lookback = text[i:end]
        boundary = max(
            lookback.rfind(". "),
            lookback.rfind("? "),
            lookback.rfind("! "),
            lookback.rfind("; "),
            lookback.rfind(": "),
        )
        if boundary != -1 and end < n and (end - (i + boundary) <= 80):
            end = i + boundary + 2  # include boundary and following space

        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break
        # advance with overlap
        i = max(0, end - overlap)
        if i >= n:
            break

    return chunks


def build_embedding_function(model_name: str):
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def add_or_replace(collection, ids: List[str], documents: List[str], metadatas: List[Dict[str, Any]], recreate: bool):
    """
    If recreate=True, collection is assumed to be new/empty and we just add.
    If recreate=False, we attempt to delete incoming ids first, then add.
    """
    if not recreate and ids:
        try:
            # delete only what we're about to write to keep operation idempotent
            collection.delete(ids=ids)
        except Exception as e:
            log(f"[warn] delete(ids=...) failed, continuing: {e!r}")
    collection.add(ids=ids, documents=documents, metadatas=metadatas)


def index(docs: List[Dict[str, Any]], size: int, overlap: int, recreate: bool) -> Tuple[int, int]:
    """
    Returns (num_pages, num_chunks)
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # create / reuse collection
    if recreate:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    try:
        col = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=build_embedding_function(EMBEDDING_MODEL),
        )
    except TypeError:
        # older client fallback signature (rare)
        col = client.get_or_create_collection(COLLECTION_NAME, embedding_function=build_embedding_function(EMBEDDING_MODEL))

    ids: List[str] = []
    passages: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for d in tqdm(docs, desc="Chunk & stage"):
        url = (d.get("canonical_url") or d.get("url") or "").strip()
        title = (d.get("title") or "").strip()
        domain = (d.get("domain") or "").strip()
        topic = (d.get("topic") or "").strip()
        text = (d.get("text") or "").strip()

        base = sha16(url or text[:200])
        chunks = smart_chunks(text, size=size, overlap=overlap)
        for idx, ch in enumerate(chunks):
            cid = f"{base}-{idx:04d}"
            ids.append(cid)
            passages.append(ch)
            metadatas.append({
                "url": url,
                "title": title,
                "domain": domain or None,
                "topic": topic or None,
                "snippet": ch[:200],
                "chunk_index": idx,
            })

    if not passages:
        log("Nothing to index (no passages).")
        return (0, 0)

    log(f"Writing to Chroma: dir={CHROMA_DIR}, collection={COLLECTION_NAME}, recreate={recreate}")
    add_or_replace(col, ids=ids, documents=passages, metadatas=metadatas, recreate=recreate)

    # stats
    try:
        stats = col.count()
        log(f"Collection count now: {stats}")
    except Exception:
        pass

    return (len(docs), len(passages))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Doc_GPT indexer → Chroma")
    p.add_argument("--min-chars", type=int, default=MIN_PAGE_CHARS_DEFAULT, help="Skip pages with fewer chars")
    p.add_argument("--size", type=int, default=CHUNK_SIZE_DEFAULT, help="Chunk size (chars)")
    p.add_argument("--overlap", type=int, default=CHUNK_OVERLAP_DEFAULT, help="Chunk overlap (chars)")
    p.add_argument("--recreate", action="store_true", help="Recreate the collection from scratch")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    docs = load_raw(min_chars=args.min_chars)
    if not docs:
        return

    pages, chunks = index(docs, size=args.size, overlap=args.overlap, recreate=args.recreate)
    log(f"Indexed {chunks} chunks from {pages} pages.")
    log(f"CHROMA_DIR={CHROMA_DIR.resolve()}")
    log(f"COLLECTION_NAME={COLLECTION_NAME}")
    log(f"EMBEDDING_MODEL={EMBEDDING_MODEL}")


if __name__ == "__main__":
    main()
