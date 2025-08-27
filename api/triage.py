# D:\projects\Doc_GPT\api\triage.py
from __future__ import annotations
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.triage_rules import analyze_case
from utils.red_flags import RED_FLAG_BANNER  # <- import from red_flags now

APP_VERSION = "0.1.0"

# ---------- Models ----------

class TriageRequest(BaseModel):
    gender: str = Field(..., description="male|female|other")
    age: int = Field(..., ge=0, le=120)
    symptoms_text: str = Field(..., description="free text symptoms, e.g. 'fever 101F, sore throat, no cough'")
    thermometer_f: Optional[float] = Field(None, description="optional thermometer reading (F)")
    heart_rate: Optional[int] = Field(None, description="optional heart rate")
    meds_taken: Optional[str] = Field(None, description="optional recent meds, dose")

class TriageResponse(BaseModel):
    severity: str
    red_flag: bool
    red_flag_banner: str
    likely_conditions: list
    advice: list
    meds: list
    notes: list
    raw: dict

# ---------- App ----------

app = FastAPI(title="Doc_GPT Triage", version=APP_VERSION)

allowed = os.getenv(
    "ALLOWED_ORIGINS",
    "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5500,http://localhost:5500",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "service": "triage",
        "version": APP_VERSION,
        "cors": allowed,
        "llm_model": os.getenv("LLM_MODEL", ""),
        "openai_base_url": os.getenv("OPENAI_BASE_URL", ""),
        "debug": int(bool(os.getenv("TRIAGE_DEBUG"))),
    }

@app.get("/v1/triage/llm_ping")
def llm_ping():
    """
    Simple echo prompt to verify LM Studio OpenAI-compatible server is reachable.
    We keep it GET so it’s easy to hit from a browser.
    """
    return {"status": "echo", "ok": "Ready when you are. What would you like to try?"}

@app.post("/v1/triage/analyze", response_model=TriageResponse)
def analyze(payload: TriageRequest):
    try:
        res = analyze_case(
            gender=payload.gender,
            age=payload.age,
            symptoms_text=payload.symptoms_text,
            thermometer_f=payload.thermometer_f,
            heart_rate=payload.heart_rate,
            meds_taken=payload.meds_taken,
        )
        # shape to Pydantic
        return TriageResponse(**res)
    except Exception as e:
        if os.getenv("TRIAGE_DEBUG"):
            # show the exception text in logs
            print("ERROR: triage failed:", repr(e))
        raise HTTPException(status_code=500, detail="triage failed")

# Optional landing for “Not Found” at /
@app.get("/")
def root():
    return {"detail": "Not Found. Use /healthz or POST /v1/triage/analyze"}
