# D:\projects\Doc_GPT\api\main.py
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from .schemas import (
    AskRequest, AskResponse, IntakeQuery, IntakeResponse, Citation, RedFlag, ConditionCandidate
)
from utils import red_flags as rf
from utils.triage_rules import feverpain_from_text, centor_from_text, guess_conditions, red_flag_assessment

from utils.first_aid import steps_for
from utils.otc import otc_suggestions
from utils.followups import follow_ups_for

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
DATA_DIR = ROOT / "data"
FLAT_DIR = Path(os.getenv("FLAT_INDEX_DIR", str(DATA_DIR / "flatindex")))
EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Minimal flat retriever ---------------------------------------------------
# expects files built by scripts/build_flat_index.py: embeddings.npy, meta.json
class FlatIndex:
    def __init__(self, path: Path):
        self.path = path
        self.ok = False
        self.emb = None  # (N, d) np.float32
        self.meta = []   # list of dicts with url, title, snippet, chunk_index
        self.model = None

        try:
            efile = self.path / "embeddings.npy"
            mfile = self.path / "meta.json"
            if not (efile.exists() and mfile.exists()):
                return
            self.emb = np.load(str(efile))
            self.meta = json.loads(mfile.read_text(encoding="utf-8"))
            # lazy load model
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMBED_MODEL_NAME)
            self.ok = True
        except Exception as e:
            print("[flat] failed to load:", repr(e))
            self.ok = False

    def search(self, query: str, k: int = 6) -> List[Tuple[float, dict]]:
        if not self.ok:
            return []
        vec = self.model.encode([query], normalize_embeddings=True)
        # cosine similarity against normalized embeddings
        E = self.emb
        q = vec[0].astype(np.float32)
        sims = (E @ q).astype(np.float32)  # (N,)
        top_idx = np.argsort(-sims)[: max(1, k)]
        out = []
        for i in top_idx.tolist():
            m = self.meta[i]
            out.append((float(sims[i]), m))
        return out

flat = FlatIndex(FLAT_DIR)
if flat.ok:
    print(f"[flat] ready: N={flat.emb.shape[0]}, dim={flat.emb.shape[1]} at {FLAT_DIR}")
else:
    print(f"[flat] NOT READY (missing {FLAT_DIR})")

# --- FastAPI app --------------------------------------------------------------
app = FastAPI(title="Doc_GPT")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# serve /web
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

# --- Small utils for composing responses -------------------------------------

def build_citations(hits: List[Tuple[float, dict]]) -> List[Citation]:
    cits: List[Citation] = []
    for i, (score, meta) in enumerate(hits, 1):
        cits.append(Citation(n=i, title=str(meta.get("title") or "Source"),
                             url=str(meta.get("url") or ""), score=round(float(score), 3)))
    return cits

def short_answer_from_docs(query: str, hits: List[Tuple[float, dict]]) -> str:
    """
    Extremely lightweight composer: pull a few useful snippets and stitch them.
    Keeps LLM optional — your LM Studio can be wired later if you want.
    """
    lines: List[str] = [f"I’ll summarise from the retrieved sources:"]
    used = 0
    for score, meta in hits[:4]:
        snip = (meta.get("snippet") or meta.get("title") or "").strip()
        if not snip:
            continue
        # basic clean
        snip = snip.replace("\n", " ")
        lines.append(f"• {snip[:320]}…")
        used += 1
    if not used:
        lines.append("• (No relevant snippet was retrieved.)")
    lines.append("")
    lines.append("Citations are listed below.")
    lines.append("—")
    lines.append("This is general information, not medical advice.")
    return "\n".join(lines)

# --- Endpoints ----------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {
        "status": "ok" if flat.ok else "index-missing",
        "backend": "flat",
        "model": EMBED_MODEL_NAME,
        "count": int(flat.emb.shape[0]) if flat.ok else 0,
        "dim": int(flat.emb.shape[1]) if flat.ok else 0,
        "dir": str(FLAT_DIR),
    }

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not flat.ok:
        raise HTTPException(status_code=503, detail="Index not ready. Build the index first.")

    hits = flat.search(req.query, k=req.k)
    cits = build_citations(hits)
    # red-flag banner
    severity, matches = ("emergency", ["emergency triggers detected"]) if rf.detect_red_flags(req.query) else ("none", [])
    banner = rf.RED_FLAG_MSG if severity != "none" else ""
    ans = short_answer_from_docs(req.query, hits)

    return AskResponse(
        answer=ans,
        citations=cits,
        red_flag=RedFlag(severity=severity, matches=matches, banner=banner),
    )

@app.post("/intake", response_model=IntakeResponse)
def intake(q: IntakeQuery):
    if not flat.ok:
        raise HTTPException(status_code=503, detail="Index not ready. Build the index first.")

    # Build a retrieval query string from structured info
    profile = q.profile
    vitals = q.vitals
    s_list = [s.strip() for s in (q.symptoms or []) if s.strip()]
    joined_symptoms = ", ".join(s_list) if s_list else "unspecified symptoms"

    parts = [
        f"Gender: {profile.gender}",
        f"Age: {profile.age}",
        f"Symptoms: {joined_symptoms}",
    ]
    if vitals and vitals.temperature_c:
        parts.append(f"T={vitals.temperature_c:.1f}C")

    query = f"Clinical triage for: " + "; ".join(parts) + f". Notes: {q.free_text or ''}"
    hits = flat.search(query, k=q.k)
    cits = build_citations(hits)

    # Condition ranking (heuristic + doc titles)
    doc_titles = [h[1].get("title") or "" for h in hits]
    candidates = guess_conditions(q.free_text, s_list, doc_titles)

    # First aid + OTC
    fa: List[str] = []
    for name, _conf in candidates[:2] or [("General health advice", 0.3)]:
        fa.extend(steps_for(name))
    otc = otc_suggestions(age=profile.age, gender=profile.gender, suspected=[c[0] for c in candidates])

    # Follow-up questions
    fups = follow_ups_for([c[0] for c in candidates])

    # Red flags
    severity, matches = red_flag_assessment(q.free_text + " " + " ".join(s_list))
    banner = rf.RED_FLAG_MSG if severity != "none" else ""

    # Compose a brief narrative answer
    top_line = "Based on your details, here’s an initial nurse‑style summary:"
    bullet_conditions = "\n".join([f"• {name} — confidence {conf:.0%}" for name, conf in candidates]) if candidates else "• No clear condition detected."
    narrative = "\n".join([
        top_line,
        bullet_conditions,
        "",
        "Immediate self‑care (first aid):",
        *[f"• {s}" for s in fa],
        "",
        "Over‑the‑counter options (if suitable for you):",
        *[f"• {s}" for s in otc],
        "",
        "Next questions I have for you:",
        *[f"• {x}" for x in fups],
        "",
        "Citations are listed below. This is general information, not medical advice."
    ])

    return IntakeResponse(
        condition_candidates=[ConditionCandidate(name=n, confidence=c) for n, c in candidates],
        first_aid=fa,
        otc=otc,
        follow_ups=fups,
        answer=narrative,
        citations=cits,
        red_flag=RedFlag(severity=severity, matches=matches, banner=banner),
    )

# Index.html convenience
@app.get("/", response_class=HTMLResponse)
def root():
    if (WEB_DIR / "index.html").exists():
        return FileResponse(str(WEB_DIR / "index.html"))
    return HTMLResponse("<h3>Doc_GPT API is running. Open /web/index.html for the UI.</h3>")
