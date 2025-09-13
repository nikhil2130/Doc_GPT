# D:\projects\Doc_GPT\api\schemas.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Core RAG API schemas
# ---------------------------------------------------------------------------

Gender = Literal["male", "female", "other", "unknown"]


class UserProfile(BaseModel):
    gender: Gender = "unknown"
    age: int = Field(ge=0, le=120)


class Vitals(BaseModel):
    temperature_c: Optional[float] = Field(default=None, ge=30, le=45)
    heart_rate: Optional[int] = Field(default=None, ge=30, le=220)
    spo2: Optional[int] = Field(default=None, ge=50, le=100)


class AskRequest(BaseModel):
    query: str = Field(..., description="User question/symptoms")
    k: int = Field(6, ge=1, le=20, description="Number of chunks to include")


class Citation(BaseModel):
    n: str
    title: str
    url: str
    score: str


class RetrievedItem(BaseModel):
    text: str
    meta: Dict[str, Any]
    score: float
    rerank_score: Optional[float] = None
    rerank_kind: Optional[str] = None


class RedFlag(BaseModel):
    severity: Literal["none", "consider", "urgent", "emergency"] = "none"
    matches: List[str] = Field(default_factory=list)
    banner: str = ""


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    red_flag: RedFlag = Field(default_factory=RedFlag)
    retrieved: Optional[List[RetrievedItem]] = None


# ---------------------------------------------------------------------------
# Intake / triage helper schemas
# ---------------------------------------------------------------------------

class IntakeQuery(BaseModel):
    profile: UserProfile
    symptoms: List[str] = Field(default_factory=list)
    free_text: str = ""
    vitals: Optional[Vitals] = None
    k: int = 6


class ConditionCandidate(BaseModel):
    name: str
    confidence: float


class IntakeResponse(BaseModel):
    condition_candidates: List[ConditionCandidate] = Field(default_factory=list)
    first_aid: List[str] = Field(default_factory=list)
    otc: List[str] = Field(default_factory=list)
    follow_ups: List[str] = Field(default_factory=list)
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    red_flag: RedFlag = Field(default_factory=RedFlag)
