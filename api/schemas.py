# D:\projects\Doc_GPT\api\schemas.py
from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

Gender = Literal["male", "female", "other", "unknown"]

class UserProfile(BaseModel):
    gender: Gender = "unknown"
    age: int = Field(ge=0, le=120)

class Vitals(BaseModel):
    temperature_c: Optional[float] = Field(default=None, ge=30, le=45)
    heart_rate: Optional[int] = Field(default=None, ge=30, le=220)
    spo2: Optional[int] = Field(default=None, ge=50, le=100)

class AskRequest(BaseModel):
    query: str
    k: int = 6

class Citation(BaseModel):
    n: int
    title: str
    url: str
    score: float

class RedFlag(BaseModel):
    severity: Literal["none", "consider", "urgent", "emergency"] = "none"
    matches: List[str] = []
    banner: str = ""

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation] = []
    red_flag: RedFlag = RedFlag()

class IntakeQuery(BaseModel):
    profile: UserProfile
    symptoms: List[str] = []
    free_text: str = ""
    vitals: Optional[Vitals] = None
    k: int = 6

class ConditionCandidate(BaseModel):
    name: str
    confidence: float

class IntakeResponse(BaseModel):
    condition_candidates: List[ConditionCandidate] = []
    first_aid: List[str] = []
    otc: List[str] = []
    follow_ups: List[str] = []
    answer: str
    citations: List[Citation] = []
    red_flag: RedFlag = RedFlag()
