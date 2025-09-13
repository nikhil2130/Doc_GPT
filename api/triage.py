# D:\projects\Doc_GPT\api\triage.py
from __future__ import annotations

import os
import logging
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Core rules/logic (no heavy imports here)
from utils.triage_rules import analyze_case

VERSION = "0.1.0"

log = logging.getLogger("triage")
logging.basicConfig(level=logging.INFO)

# --- Env (must match LM Studio / your local model) ---
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_MODEL       = os.getenv("LLM_MODEL", "meta-llama-3.1-8b-instruct")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "lm-studio")  # any non-empty str is fine for LM Studio
TRIAGE_DEBUG    = os.getenv("TRIAGE_DEBUG", "0") == "1"

# --- FastAPI app ---
app = FastAPI(title="Doc_GPT Triage Service", version=VERSION)

# CORS: allow web client + your dev ports
_allowed = {
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_allowed),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---
class TriageInput(BaseModel):
    gender: str = Field(..., description="male | female | other")
    age: int = Field(..., ge=0, le=120)
    symptoms_text: str = Field(..., description="free text symptoms (e.g., 'fever 101F, sore throat, no cough')")
    thermometer_f: Optional[float] = Field(None, description="Temperature in Fahrenheit if known")
    heart_rate: Optional[int] = Field(None, description="Beats per minute if known")
    meds_taken: Optional[str] = Field(None, description="Any meds already taken (e.g., 'ibuprofen 200mg')")


class TriageOutput(BaseModel):
    severity: str
    red_flag: bool
    red_flag_banner: str
    likely_conditions: List[Dict[str, Any]]
    advice: List[str]
    meds: List[str]
    notes: Dict[str, Any]
    raw: Dict[str, Any]


# --- Routes ---
@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "service": "triage",
        "version": VERSION,
        "cors": list(_allowed),
        "llm_model": LLM_MODEL,
        "openai_base_url": OPENAI_BASE_URL,
        "debug": 1 if TRIAGE_DEBUG else 0,
    }


@app.get("/")
def root():
    return {"detail": "Not Found"}


@app.get("/v1/triage/llm_ping")
def llm_ping():
    # Lightweight echo—good for verifying LM Studio connectivity from this process
    try:
        from openai import OpenAI
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
        msg = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word: echo"}],
            max_tokens=8,
            temperature=0,
        )
        return {"status": "ok", "echo": msg.choices[0].message.content.strip()}
    except Exception as e:
        if TRIAGE_DEBUG:
            log.exception("LLM ping failed")
        raise HTTPException(status_code=500, detail=f"llm ping failed: {e!s}")


@app.post("/v1/triage/analyze", response_model=TriageOutput)
def analyze(payload: TriageInput):
    """
    Calls the rule/LLM hybrid analyzer and returns a compact, UI-friendly result.
    """
    try:
        result = analyze_case(
            gender=payload.gender,
            age=payload.age,
            symptoms_text=payload.symptoms_text,
            thermometer_f=payload.thermometer_f,
            heart_rate=payload.heart_rate,
            meds_taken=payload.meds_taken,
            openai_base_url=OPENAI_BASE_URL,
            llm_model=LLM_MODEL,
            openai_api_key=OPENAI_API_KEY,
            debug=TRIAGE_DEBUG,
        )
        return result
    except Exception as e:
        if TRIAGE_DEBUG:
            log.exception("triage failed")
        raise HTTPException(status_code=500, detail="triage failed")
