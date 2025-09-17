# api/main.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sklearn.neighbors import NearestNeighbors
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tenacity import retry, wait_exponential, stop_after_attempt

# OpenAI-compatible client (for LM Studio or OpenAI)
try:
    from openai import OpenAI  # openai>=1.x
except Exception:  # fallback if older SDK is present
    OpenAI = None

# Local utils
from utils.red_flags import detect_red_flags, classify_red_flag_severity, RED_FLAG_BANNER
from .config import get_settings, iter_existing_env_files

settings = get_settings()
ENV_FILES_LOADED = tuple(str(path) for path in iter_existing_env_files())


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
FLAT_DIR = settings.flat_index_dir
WEB_DIR = settings.web_dir
EMB_MODEL_NAME = settings.embedding_model

# Hybrid (dense + BM25) controls
HYBRID = True
ALPHA = max(0.0, min(1.0, settings.hybrid_alpha))  # weight for dense side in [0,1]

# Reranker (optional)
RERANK = True
RERANK_MODEL = settings.rerank_model
RERANK_TOP_M = settings.rerank_top_m

# LLM (LM Studio or OpenAI-compatible)
OPENAI_BASE_URL = settings.openai_base_url
OPENAI_API_KEY = settings.openai_api_key  # any non-empty string for LM Studio
LLM_MODEL = settings.llm_model

# Misc
SHOW_RETRIEVED = settings.show_retrieved


# --------------------------------------------------------------------------------------
# Data Models
# --------------------------------------------------------------------------------------
from .schemas import AskRequest, AskResponse, Citation, RetrievedItem, RedFlag


# --------------------------------------------------------------------------------------
# Load flat index
#   - embeddings.npy: dense vectors (N, dim)
#   - meta.json     : list of {text, url, title, ...} for each chunk
#   - nn.joblib     : (optional) prebuilt NearestNeighbors index (not required)
# --------------------------------------------------------------------------------------
def _load_flat(dir_path: Path) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    if not dir_path.exists():
        raise RuntimeError(f"Flat index dir not found: {dir_path}")

    emb_path = dir_path / "embeddings.npy"
    meta_path = dir_path / "meta.json"

    if not emb_path.exists() or not meta_path.exists():
        raise RuntimeError("Index not ready. Build the index first.")

    embeddings = np.load(emb_path)
    meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))

    # `meta.json` used to be a dict with separate "texts" and "metas" lists.
    # Newer versions store a list of chunk dictionaries. Handle both for
    # backward compatibility.
    if isinstance(meta_raw, dict) and "texts" in meta_raw and "metas" in meta_raw:
        texts = meta_raw.get("texts", [])
        metas = meta_raw.get("metas", [])
        meta: List[Dict[str, Any]] = []
        for t, m in zip(texts, metas):
            md = m if isinstance(m, dict) else {}
            md = {**md, "text": t}
            meta.append(md)
    elif isinstance(meta_raw, list):
        meta = meta_raw
    else:
        raise RuntimeError("Unrecognized meta.json format")

    return embeddings, meta


def _build_bm25(corpus: List[str]) -> BM25Okapi:
    # very simple whitespace tokenization
    tokenized = [doc.lower().split() for doc in corpus]
    return BM25Okapi(tokenized)


# --------------------------------------------------------------------------------------
# RAG components
# --------------------------------------------------------------------------------------
class RAGEngine:
    def __init__(self, flat_dir: Path):
        self.dir = flat_dir
        self.embeddings, self.meta = _load_flat(flat_dir)
        self.N, self.dim = self.embeddings.shape
        # Embedding model
        self.emb_model = SentenceTransformer(EMB_MODEL_NAME)
        # Dense index
        self.nn = NearestNeighbors(metric="cosine", algorithm="auto")
        self.nn.fit(self.embeddings)
        # BM25
        self.texts = [m.get("text") or m.get("snippet") or "" for m in self.meta]
        self.bm25 = _build_bm25(self.texts) if HYBRID else None

        # Optional reranker
        self.reranker = None
        if RERANK:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(RERANK_MODEL)
                self.rerank_kind = "ce"
            except Exception:
                self.reranker = None
                self.rerank_kind = None

        # LLM client
        self.llm = None
        if OpenAI is not None:
            try:
                self.llm = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
            except Exception:
                self.llm = None

    def status(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "backend": "flat",
            "mode": "hybrid" if HYBRID else "dense",
            "alpha": ALPHA,
            "reranker": "enabled" if self.reranker else "disabled",
            "rerank_model": RERANK_MODEL if self.reranker else "",
            "rerank_top_m": RERANK_TOP_M if self.reranker else 0,
            "model": EMB_MODEL_NAME,
            "count": self.N,
            "dim": self.dim,
            "dir": str(self.dir.resolve()),
            "bm25": bool(self.bm25),
        }

    def embed_query(self, q: str) -> np.ndarray:
        v = self.emb_model.encode([q], normalize_embeddings=True)
        return v[0]

    def search(self, q: str, k: int) -> List[Tuple[int, float]]:
        """Return [(idx, score)] of top-k by hybrid score."""
        qv = self.embed_query(q)

        # Dense (cosine sim = 1 - distance)
        dists, idxs = self.nn.kneighbors([qv], n_neighbors=min(max(k * 2, 10), self.N))
        idxs = idxs[0]
        sims = 1.0 - dists[0]  # convert cosine distance to similarity

        dense_scores = {int(i): float(s) for i, s in zip(idxs, sims)}

        if self.bm25 is not None:
            bm25_scores = self.bm25.get_scores(q.lower().split())
            # normalize both sides to [0,1]
            dvals = np.array(list(dense_scores.values()))
            dnorm = (dvals - dvals.min()) / (dvals.ptp() + 1e-9)

            bvals = np.array(bm25_scores)
            bnorm = (bvals - bvals.min()) / (bvals.ptp() + 1e-9)

            # hybrid score
            combined: Dict[int, float] = {}
            for i, _ in enumerate(self.meta):
                ds = dnorm[list(dense_scores.keys()).index(i)] if i in dense_scores else 0.0
                bs = float(bnorm[i])
                combined[i] = ALPHA * ds + (1.0 - ALPHA) * bs
            # sort
            top = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
            return top
        else:
            # dense only
            top = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)[:k]
            return top

    def rerank(self, q: str, candidates: List[Tuple[int, float]]) -> List[Tuple[int, float, float]]:
        """Return [(idx, hybrid_score, rerank_score)]"""
        if not self.reranker:
            return [(i, s, None) for i, s in candidates]

        pair_inputs = [(q, self.meta[i].get("text") or self.texts[i]) for i, _ in candidates[:RERANK_TOP_M]]
        scores = self.reranker.predict(pair_inputs).tolist()
        out = []
        for (i, s), rr in zip(candidates[:RERANK_TOP_M], scores):
            out.append((i, s, float(rr)))
        # keep ordering by rerank score, then hybrid score
        out.sort(key=lambda x: (x[2], x[1]), reverse=True)
        return out[: len(candidates)]

    @retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3))
    def call_llm(self, question: str, contexts: List[Dict[str, str]]) -> str:
        """Call a local LLM through OpenAI-compatible API to synthesize the answer."""
        if not self.llm:
            # best-effort fallback if no client
            joined = "\n\n".join([f"[{i+1}] {c['title']} — {c['url']}\n{c['text']}" for i, c in enumerate(contexts)])
            return f"I'll summarize from the provided sources:\n\n{joined}\n\n(Citations above.)\n\n—\nThis is general information, not medical advice. For personal guidance, consult a clinician."

        sys_prompt = (
            "You are Doc_GPT, a careful medical assistant. Use ONLY the provided snippets. "
            "Write concise, plain-English guidance. End with a short disclaimer. "
            "When you reference a source, place a bracketed number like [1], [2], ... that matches the provided context list."
        )

        ctx_lines = []
        for i, c in enumerate(contexts, start=1):
            ctx_lines.append(f"[{i}] {c['title']} | {c['url']}\n{c['text']}")
        ctx_block = "\n\n".join(ctx_lines)

        user_prompt = (
            f"Question: {question}\n\n"
            f"Context (numbered):\n{ctx_block}\n\n"
            "Instructions:\n"
            "- Synthesize an answer grounded in the snippets above.\n"
            "- Include bracketed citation numbers (e.g., [1], [2]) wherever claims are supported.\n"
            "- If red-flag symptoms are present in the question, remind the user to seek urgent care.\n"
            "- Finish with: “—\\nThis is general information, not medical advice. For personal guidance, consult a clinician.”"
        )

        resp = self.llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        return resp.choices[0].message.content or ""

    def answer(self, q: str, k: int) -> AskResponse:
        # 1) retrieve
        raw = self.search(q, k=max(k * 2, 10))
        ranked = self.rerank(q, raw)
        top = ranked[:k]

        # 2) build contexts & citations
        contexts: List[Dict[str, str]] = []
        citations: List[Citation] = []
        retrieved: List[RetrievedItem] = []

        for i, (idx, score, rr) in enumerate(top, start=1):
            m = self.meta[idx]
            title = (m.get("title") or "").strip() or "Untitled"
            url = m.get("url") or ""
            text = m.get("text") or m.get("snippet") or ""
            contexts.append({"title": title, "url": url, "text": text})
            citations.append(Citation(n=str(i), title=title, url=url, score=f"{score:.3f}"))
            retrieved.append(
                RetrievedItem(
                    text=text,
                    meta={"url": url, "title": title, "domain": m.get("domain"), "topic": m.get("topic"), "snippet": m.get("snippet"), "chunk_index": m.get("chunk_index")},
                    score=float(score),
                    rerank_score=float(rr) if rr is not None else None,
                    rerank_kind=getattr(self, "rerank_kind", None),
                )
            )

        # 3) generate draft with LLM
        draft = self.call_llm(q, contexts)

        # 4) red-flag pass
        red = RedFlag(
            severity=classify_red_flag_severity(q),
            matches=[],
            banner=RED_FLAG_BANNER if detect_red_flags(q) else "",
        )

        # 5) return
        return AskResponse(
            answer=draft,
            citations=citations,
            red_flag=red,
            retrieved=retrieved if SHOW_RETRIEVED else None,
        )


# --------------------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------------------
app = FastAPI(title="Doc_GPT API", version="0.1.0")

if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

# CORS: allow local UI/dev ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redirect root to the web UI when available
@app.get("/", include_in_schema=False)
def serve_root() -> Any:
    if WEB_DIR.exists():
        return RedirectResponse(url="/web/index.html")
    return {"status": "ok"}

# Initialize engine at import time
try:
    ENGINE = RAGEngine(FLAT_DIR)
except Exception as e:
    ENGINE = None
    INIT_ERR = e
else:
    INIT_ERR = None


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    env_info = {
        "env_files": list(ENV_FILES_LOADED),
        "web_dir": str(WEB_DIR),
        "web_available": WEB_DIR.exists(),
    }

    if ENGINE is None:
        return {
            "status": "error",
            "detail": str(INIT_ERR) if INIT_ERR else "unknown",
            **env_info,
        }

    payload = ENGINE.status()
    payload.update(env_info)
    return payload


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Index not ready. Build the index first.")
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=422, detail="Missing 'query'")
    try:
        return ENGINE.answer(req.query.strip(), req.k)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Request failed: {e!r}")
