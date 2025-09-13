# utils/triage_rules.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# Our red flags helper (you already have this file)
from .red_flags import detect_red_flags, RED_FLAG_BANNER

# -----------------------------
# Small parsing helpers
# -----------------------------
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

def _to_float(val: Any) -> float | None:
    """
    Best-effort float parsing from free text or numbers.
    Examples that become 101.4: "101.4", "Temp=101.4F", "fever 101.4 F".
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def _has(token: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(token)}\b", text, flags=re.I) is not None

def _any(tokens: List[str], text: str) -> bool:
    return any(_has(t, text) for t in tokens)


# -----------------------------
# Symptom scores (very lightweight heuristics)
# -----------------------------
def feverpain_from_text(text: str) -> Tuple[int, Dict[str, Any]]:
    """
    FEVERPAIN heuristic (0–5). We don’t try to be perfect—just extract a few
    common cues from text to demonstrate the flow.
    Score +1 each:
      - Fever (we treat >=100.4F as fever if we find a number with F/°F or the word 'fever')
      - Absence of cough
      - Rapid onset / severe tonsillar pain (keywords)
      - Pus/exudate words
      - Inflamed/tender lymph nodes (neck glands)
    """
    t = text.lower()
    score = 0
    notes = {}

    # Fever: explicit number like 101F or keyword "fever"
    has_fever_kw = "fever" in t
    temp_matches = re.findall(r"(\d{2,3}(?:\.\d+)?)\s*°?\s*f", t)
    tempF = float(temp_matches[0]) if temp_matches else None
    fever_by_temp = tempF is not None and tempF >= 100.4
    if has_fever_kw or fever_by_temp:
        score += 1
        notes["fever"] = tempF if tempF is not None else True

    # Absence of cough
    no_cough = ("no cough" in t) or ("without cough" in t) or ("absent cough" in t)
    if no_cough:
        score += 1
        notes["no_cough"] = True

    # Rapid onset / severe pain
    if _any(["rapid", "sudden", "severe pain", "very sore", "very painful"], t):
        score += 1
        notes["rapid_or_severe"] = True

    # Exudate / pus
    if _any(["exudate", "pus", "white patches", "tonsillar exudate"], t):
        score += 1
        notes["exudate"] = True

    # Tender glands
    if _any(["tender glands", "swollen glands", "neck lymph", "lymph node"], t):
        score += 1
        notes["lymph_nodes"] = True

    return score, notes


def centor_from_text(text: str, age: int | None = None) -> Tuple[int, Dict[str, Any]]:
    """
    Centor (0–4) – rough text cues:
      +1: tonsillar exudates
      +1: tender anterior cervical nodes
      +1: fever
      +1: absence of cough
    (Age adjustments are handled downstream as +1 / 0 / -1 in many variants; here
     we keep raw 0–4 and only report the age band in notes.)
    """
    t = text.lower()
    score = 0
    notes = {}

    if _any(["exudate", "pus", "white patches", "tonsillar exudate"], t):
        score += 1
        notes["exudate"] = True

    if _any(["tender glands", "tender nodes", "anterior cervical", "neck lymph"], t):
        score += 1
        notes["tender_nodes"] = True

    if "fever" in t or re.search(r"\b(10[0-9](?:\.\d+)?)\s*°?\s*f\b", t):
        score += 1
        notes["fever"] = True

    if ("no cough" in t) or ("without cough" in t) or ("absent cough" in t):
        score += 1
        notes["no_cough"] = True

    if age is not None:
        if age < 15:
            notes["age_band"] = "<15"
        elif age <= 44:
            notes["age_band"] = "15–44"
        else:
            notes["age_band"] = "≥45"

    return score, notes


# -----------------------------
# Condition guessing (toy probabilities)
# -----------------------------
def guess_conditions(fp_score: int, centor_score: int) -> List[Dict[str, Any]]:
    """
    Produce a tiny ranked list with toy probabilities for demo purposes.
    Higher FEVERPAIN / Centor nudges toward strep.
    """
    # crude mapping
    strep_p = 0.10 + 0.10 * fp_score + 0.10 * centor_score
    strep_p = max(0.0, min(0.90, strep_p))
    viral_p = max(0.0, 1.0 - strep_p)

    return [
        {"name": "Streptococcal pharyngitis (strep throat)", "p": round(strep_p, 2)},
        {"name": "Viral pharyngitis (common cold/flu)", "p": round(viral_p, 2)},
        {"name": "Other upper-respiratory causes", "p": round(1.0 - (strep_p + viral_p), 2)},
    ]


def first_aid_and_meds(age: int | None, text: str) -> Tuple[List[str], List[str]]:
    """
    Very conservative, OTC-only suggestions. This is **not** medical advice.
    """
    advice = [
        "Rest and hydrate (warm fluids can soothe the throat).",
        "Gargle warm salt water (not for young children).",
        "Avoid smoking/smoky places; consider honey/lemon for cough or throat irritation (not for <1 year old).",
        "Seek urgent care if symptoms are severe, rapidly worsening, or new red flags appear."
    ]
    meds = [
        "Paracetamol/acetaminophen for fever/pain (follow label dosing).",
        "Ibuprofen if tolerated (with food); avoid if you have ulcers or kidney problems.",
        "Throat lozenges or sprays for symptomatic relief (age-appropriate)."
    ]
    return advice, meds


# -----------------------------
# Public API
# -----------------------------
def analyze_case(**payload: Any) -> Dict[str, Any]:
    """
    Main entrypoint used by api/triage.py.
    Accepts flexible keyword args:
      - gender: str | None
      - age: int | None
      - symptoms_text: str  (required)
      - thermometer_f: float | str | None
      - heart_rate: int | str | None
      - meds_taken: str | None
      - (and any other fields; safely ignored)
    Returns a dict safe to JSON-serialize.
    """
    gender = (payload.get("gender") or "").strip().lower() or None
    age = payload.get("age")
    try:
        age = int(age) if age is not None else None
    except Exception:
        age = None

    text = payload.get("symptoms_text") or payload.get("text") or ""
    text = str(text)

    tempF = payload.get("thermometer_f")
    tempF = _to_float(tempF)

    heart_rate_raw = payload.get("heart_rate")
    try:
        heart_rate = int(heart_rate_raw) if heart_rate_raw is not None else None
    except Exception:
        heart_rate = None

    meds_taken = payload.get("meds_taken")
    meds_taken = str(meds_taken) if meds_taken is not None else None

    # Compute basic scores
    fp, fp_notes = feverpain_from_text(text)
    centor, centor_notes = centor_from_text(text, age=age)

    # Red flags
    flags = detect_red_flags(text)
    red_flag = len(flags) > 0
    severity = "emergency" if red_flag else "none"

    # Guess conditions & basic advice
    likely = guess_conditions(fp, centor)
    advice, meds = first_aid_and_meds(age, text)

    # Build response
    out: Dict[str, Any] = {
        "severity": severity,
        "red_flag": red_flag,
        "red_flag_banner": RED_FLAG_BANNER if red_flag else "",
        "likely_conditions": likely,
        "advice": advice,
        "meds": meds,
        "notes": {
            "FEVERPAIN": fp,
            "CENTOR": centor,
            "TempF": tempF,
            "heart_rate": heart_rate,
            "meds_taken": meds_taken,
        },
        "raw": {
            "feverpain": fp_notes,
            "centor": centor_notes,
        },
    }
    return out
