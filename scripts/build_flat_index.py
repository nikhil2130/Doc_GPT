# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import joblib

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "flatindex"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE_DEFAULT = 800
CHUNK_OVERLAP_DEFAULT = 120
MIN_PAGE_CHARS_DEFAULT = 400

def log(msg: str) -> None:
    print(msg, flush=True)

def smart_chunks(text: str, size: int, overlap: int) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    n = len(text)
    if n == 0:
        return []
    chunks: List[str] = []
    i = 0
    while i < n:
        end = min(i + size, n)
        lookback = text[i:end]
        boundary = max(
            lookback.rfind(". "),
            lookback.rfind("? "),
            lookback.rfind("! "),
            lookback.rfind("; "),
            lookback.rfind(": "),
        )
        if boundary != -1 and end < n and (end - (i + boundary) <= 80):
            end = i + boundary + 2
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n: break
        i = max(0, end - overlap)
        if i >= n: break
    return chunks

def load_pages(min_chars: int) -> List[Dict[str, Any]]:
    paths = sorted(RAW.glob("*.json"))
    if not paths:
        log("No raw pages found. Run scripts/crawl_clean.py first.")
        return []
    pages: List[Dict[str, Any]] = []
    for p in tqdm(paths, desc="Load raw JSON"):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            text = (j.get("text") or "").strip()
            if len(text) >= min_chars:
                pages.append(j)
        except Exception as e:
            log(f"[skip] bad JSON {p.name}: {e!r}")
    return pages

def build_index(pages: List[Dict[str, Any]], size: int, overlap: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    docs: List[str] = []
    meta: List[Dict[str, Any]] = []
    for d in tqdm(pages, desc="Chunk pages"):
        url = (d.get("canonical_url") or d.get("url") or "").strip()
        title = (d.get("title") or "").strip()
        domain = (d.get("domain") or "").strip()
        topic = (d.get("topic") or "").strip()
        text = (d.get("text") or "").strip()
        chunks = smart_chunks(text, size=size, overlap=overlap)
        for idx, ch in enumerate(chunks):
            docs.append(ch)
            meta.append({
                "url": url,
                "title": title,
                "domain": domain or None,
                "topic": topic or None,
                "snippet": ch[:200],
                "chunk_index": idx,
            })
    if not docs:
        return np.zeros((0, 384), dtype="float32"), meta

    log(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    emb = model.encode(docs, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
    emb = emb.astype("float32")
    return emb, meta

def save_flatindex(emb: np.ndarray, meta: List[Dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "embeddings.npy", emb)
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # Fit a NearestNeighbors index (cosine distance) and persist with joblib
    if len(emb) > 0:
        nn = NearestNeighbors(n_neighbors=50, metric="cosine")
        nn.fit(emb)
        joblib.dump(nn, OUT_DIR / "nn.joblib")
    log(f"Saved flat index to {OUT_DIR}")

def main():
    ap = argparse.ArgumentParser(description="Build flat (pure-Python) vector index")
    ap.add_argument("--size", type=int, default=CHUNK_SIZE_DEFAULT)
    ap.add_argument("--overlap", type=int, default=CHUNK_OVERLAP_DEFAULT)
    ap.add_argument("--min-chars", type=int, default=MIN_PAGE_CHARS_DEFAULT)
    args = ap.parse_args()

    pages = load_pages(min_chars=args.min_chars)
    if not pages:
        return
    emb, meta = build_index(pages, size=args.size, overlap=args.overlap)
    save_flatindex(emb, meta)
    log(f"Indexed {len(meta)} chunks from {len(pages)} pages. Embedding dim={emb.shape[1] if emb.size else 0}")

if __name__ == "__main__":
    main()
