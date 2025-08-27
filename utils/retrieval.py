from __future__ import annotations
import json, math, os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# --- helpers -----------------------------------------------------

def cosine(a, b):
    import numpy as np
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float((a @ b) / (na * nb))

@dataclass
class Passage:
    text: str
    url: str
    title: str

# --- retriever ---------------------------------------------------

class HybridRetriever:
    def __init__(self,
                 chroma_path: str,
                 collection_name: str,
                 raw_dir: str,
                 top_k_dense: int = 6,
                 top_k_bm25: int = 10,
                 mix_top_k: int = 8):
        self.top_k_dense = top_k_dense
        self.top_k_bm25 = top_k_bm25
        self.mix_top_k = mix_top_k

        # dense embeddings model (fast, good quality)
        # all-MiniLM-L6-v2: general-purpose sentence encoder (256 WP limit)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # Hugging Face docs: model usage notes & speed/quality tradeoffs.  :contentReference[oaicite:2]{index=2}

        # connect to Chroma (persistent)
        self.client = chromadb.PersistentClient(path=chroma_path, settings=Settings(allow_reset=False))
        names = [c.name for c in self.client.list_collections()]
        if collection_name not in names and names:
            collection_name = names[0]  # pick the first available if our default isn't present
        self.col = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

        # lightweight BM25 over raw docs (lexical)
        # Load raw JSON docs and split into small chunks (simple heuristic).
        self.passages: List[Passage] = []
        rawp = Path(raw_dir)
        for fp in rawp.glob("*.json"):
            obj = json.loads(fp.read_text(encoding="utf-8"))
            url = obj.get("url",""); title = obj.get("title",""); text = obj.get("text","")
            # naive chunking ~ 60-120 words
            words = text.split()
            step = 80
            for i in range(0, len(words), step):
                chunk = " ".join(words[i:i+step])
                if len(chunk) > 200:
                    self.passages.append(Passage(chunk, url, title))
        if not self.passages:
            raise RuntimeError(f"No passages constructed from {raw_dir}")

        # Build BM25
        tokenized = [p.text.lower().split() for p in self.passages]
        self.bm25 = BM25Okapi(tokenized)
        self._tokenized = tokenized  # keep for scoring & snippets
        # Rank-BM25 reference. :contentReference[oaicite:3]{index=3}

    def _dense(self, query: str, k: int) -> List[Dict[str, Any]]:
        qvec = self.model.encode([query], normalize_embeddings=True)[0]
        # query Chroma for candidate docs; we request text + meta + embeddings to fuse with cosine
        res = self.col.query(query_embeddings=[qvec.tolist()], n_results=max(k, k+4), include=["documents","metadatas","embeddings"])
        outs = []
        for doc, meta, emb in zip(res["documents"][0], res["metadatas"][0], res["embeddings"][0]):
            outs.append({
                "text": doc, "url": meta.get("url",""), "title": meta.get("title",""),
                "score_dense": cosine(qvec, emb)
            })
        outs.sort(key=lambda x: x["score_dense"], reverse=True)
        return outs[:k]

    def _bm25(self, query: str, k: int) -> List[Dict[str, Any]]:
        q_tokens = query.lower().split()
        scores = self.bm25.get_scores(q_tokens)
        ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        outs = []
        for i in ids:
            p = self.passages[i]
            outs.append({
                "text": p.text, "url": p.url, "title": p.title, "score_bm25": float(scores[i])
            })
        return outs

    def _snippet(self, txt: str, q: str, n=30) -> str:
        ql = q.lower().split()
        words = txt.split()
        for i, w in enumerate(words):
            if any(w.lower().startswith(t[:4]) for t in ql):
                a = max(0, i-n//2); b = min(len(words), i+n//2)
                return " ".join(words[a:b])
        return " ".join(words[:n])

    def search(self, query: str, profile: str = "", history: List[str] = []) -> List[Dict[str, Any]]:
        # augment query lightly with profile + last turn for recall
        aug = query
        if profile: aug += f" | profile: {profile}"
        if history: aug += f" | last: {history[-1]}"

        dense = self._dense(aug, self.top_k_dense)
        lex   = self._bm25(aug, self.top_k_bm25)

        # normalize & fuse scores (min-max)
        def norm(vals):
            if not vals: return []
            v = [x for x in vals]
            lo = min(v); hi = max(v)
            return [(x - lo) / (hi - lo + 1e-6) for x in v]

        for lst, key in [(dense, "score_dense"), (lex, "score_bm25")]:
            scores = norm([d.get(key, 0.0) for d in lst])
            for d, s in zip(lst, scores): d[key] = s

        # unified pool keyed by (url,text) to dedupe, and fuse with weighted sum
        pool: Dict[str, Dict[str, Any]] = {}
        def add(item):
            k = item["url"] + "||" + item["text"][:100]
            pool.setdefault(k, {"text": item["text"], "url": item["url"], "title": item.get("title",""),
                                "score_dense": 0.0, "score_bm25": 0.0})
            for key in ("score_dense","score_bm25"):
                if key in item: pool[k][key] = max(pool[k][key], item[key])

        for it in dense: add(it)
        for it in lex: add(it)

        # weight dense slightly higher; BM25 rescues exact terms (medication names, symptoms)
        for k, v in pool.items():
            v["score"] = 0.6 * v["score_dense"] + 0.4 * v["score_bm25"]

        ranked = sorted(pool.values(), key=lambda x: x["score"], reverse=True)[:self.mix_top_k]

        # attach short snippet for answer assembly
        out = []
        for r in ranked:
            out.append({
                "url": r["url"],
                "title": r["title"] or "NHS / CDC",
                "snippet": self._snippet(r["text"], query, n=36)
            })
        return out
