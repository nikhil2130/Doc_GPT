"""Build a compact "flat" index into ``data/flatindex``.

Artifacts:

* ``embeddings.npy`` – float32 array ``[N, 384]`` containing chunk embeddings.
* ``meta.json``      – list of dictionaries ``[{text,url,title,snippet,...}]``
* ``nn.joblib``      – ``sklearn`` ``NearestNeighbors`` (cosine) index.

Source docs come from ``data/raw/*.json`` produced by ``scripts/crawl_clean.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import joblib

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "flatindex"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

def chunk_text(t: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    t = " ".join((t or "").split())
    chunks = []
    i = 0
    while i < len(t):
        chunks.append(t[i:i+size])
        i += size - overlap
    return chunks

def load_raw_pages() -> List[Dict[str, Any]]:
    pages = []
    for p in sorted(RAW.glob("*.json")):
        try:
            pages.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"[warn] skip {p.name}: {e}")
    return pages

def main():
    pages = load_raw_pages()
    if not pages:
        print("[fatal] no raw pages found; run scripts/crawl_clean.py first")
        return

    OUT.mkdir(parents=True, exist_ok=True)

    chunks: List[Dict[str, Any]] = []
    for pg in tqdm(pages, desc="Chunk pages"):
        url = pg.get("url", "")
        title = pg.get("title", "")
        for ch in chunk_text(pg.get("text", "")):
            chunks.append(
                {
                    "text": ch,
                    "url": url,
                    "title": title,
                    "snippet": ch[:200],
                }
            )

    print(f"Loading embedding model: {EMBED_MODEL}")
    enc = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    X = enc.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    # save artifacts
    np.save(OUT / "embeddings.npy", X)
    (OUT / "meta.json").write_text(
        json.dumps(chunks, ensure_ascii=False),
        encoding="utf-8",
    )

    nn = NearestNeighbors(metric="cosine", n_neighbors=min(50, max(10, X.shape[0])))
    nn.fit(X)
    joblib.dump(nn, OUT / "nn.joblib")

    print(f"Saved flat index to {OUT}")
    print(
        f"Indexed {len(chunks)} chunks from {len(pages)} pages. Embedding dim={X.shape[1]}"
    )

if __name__ == "__main__":
    main()
