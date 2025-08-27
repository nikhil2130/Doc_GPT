# D:\projects\Doc_GPT\utils\triage_rules.py
from __future__ import annotations
import math
import re
from typing import Dict, List, Tuple, Optional

from .red_flags import (
    detect_red_flags,
    classify_red_flag_severity,
    RED_FLAG_BANNER,   # <— use this instead of the old RED_FLAG_MSG
)

_TEMP_F_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*F\b", re.I)
_TEMP_C_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*C\b", re.I)

def _parse_temperature(text: str) -> Optional[float]:
    """
    Return temperature in Fahrenheit if found, else None.
    Accepts “101F” or “38.5C”.
    """
    t = text or ""
    m = _TEMP_F_PATTERN.search(t)
    if m:
        return float(m.group(1))
    m = _TEMP_C_PATTERN.search(t)
    if m:
        c = float(m.group(1))
        return c * 9.0 / 5.0 + 32.0
    return None

# --- Simple clinical scores from plain text (heuristic) ----------------------

def feverpain_from_text(text: str) -> int:
    """
    Very soft FEVERPAIN approximation from free text:
      F (fever), P (rapidly within 3 days – ignored here), A (puss/tonsils – ignored),
      I (severely Inflamed tonsils – proxy: 'very sore throat'),
      N (No cough)
    We only score fever and absence of cough reliably from free text.
    """
    t = (text or "").lower()
    score = 0
    if "fever" in t or _parse_temperature(t) and _parse_temperature(t) >= 100.4:
        score += 1
    if "no cough" in t or ("cough" not in t and "no" in t):
        score += 1
    if "very sore throat" in t or "severe sore throat" in t or "pain swallowing" in t:
        score += 1
    return min(score, 5)

def centor_from_text(text: str, age: Optional[int] = None) -> int:
    """
    Centor (simplified):
      +1 tonsillar exudate (proxy: 'white spots', 'pus')
      +1 tender anterior cervical nodes (proxy: 'swollen glands')
      +1 fever > 38C/100.4F (from text/temperature)
      +1 absence of cough
    McIsaac age adjustment:
      3–14: +1, 15–44: 0, >=45: -1
    """
    t = (text or "").lower()
    score = 0
    if "white spots" in t or "exudate" in t or "pus" in t:
        score += 1
    if "swollen glands" in t or "tender glands" in t or "neck glands" in t:
        score += 1
    temp_f = _parse_temperature(t)
    if "fever" in t or (temp_f is not None and temp_f >= 100.4):
        score += 1
    if "no cough" in t or ("cough" not in t and "no" in t):
        score += 1

    if age is not None:
        if 3 <= age <= 14:
            score += 1
        elif age >= 45:
            score -= 1
    return score

# --- Condition guesser -------------------------------------------------------

def guess_conditions(
    text: str,
    age: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    Extremely lightweight condition prior. Returns list of (name, probability).
    We use FEVERPAIN and CENTOR to bias between viral vs strep pharyngitis.
    """
    t = (text or "").lower()
    fp = feverpain_from_text(t)
    ce = centor_from_text(t, age=age)

    # Default priors
    p_viral = 0.6
    p_strep = 0.3
    p_other = 0.1

    # Tilt with scores
    if ce >= 3 or fp >= 4:
        p_strep += 0.25
        p_viral -= 0.20
    elif ce <= 1 and fp <= 1:
        p_viral += 0.20
        p_strep -= 0.15

    # Clamp and renormalize
    p_viral = max(0.0, min(1.0, p_viral))
    p_strep = max(0.0, min(1.0, p_strep))
    p_other = max(0.0, min(1.0, 1.0 - (p_viral + p_strep)))
    Z = p_viral + p_strep + p_other
    if Z <= 0:
        p_viral, p_strep, p_other = 0.6, 0.3, 0.1
        Z = 1.0

    return [
        ("Viral pharyngitis (common cold/flu)", round(p_viral / Z, 3)),
        ("Streptococcal pharyngitis (strep throat)", round(p_strep / Z, 3)),
        ("Other upper-respiratory causes", round(p_other / Z, 3)),
    ]

# --- Red-flag wrapper ---------------------------------------------------------

def red_flag_assessment(text: str) -> Dict[str, str]:
    """
    Return { present: bool, severity: 'none'|'urgent'|'emergency', banner: str }
    """
    present = detect_red_flags(text or "")
    if not present:
        return {"present": False, "severity": "none", "banner": ""}
    severity = classify_red_flag_severity(text or "")
    banner = RED_FLAG_BANNER
    return {"present": True, "severity": severity, "banner": banner}

# --- Main engine used by api/triage.py ---------------------------------------

def analyze_case(
    gender: str,
    age: int,
    symptoms_text: str,
    thermometer_f: Optional[float] = None,
    heart_rate: Optional[int] = None,
    meds_taken: Optional[str] = None,
) -> Dict:
    """
    Pure-python rules layer (fast, deterministic).
    The API layer may optionally add an LLM step on top, but this function
    must be importable without touching OpenAI/LM Studio.
    """
    age = int(age) if age is not None else None
    text = symptoms_text or ""

    # Scores
    fp = feverpain_from_text(text)
    ce = centor_from_text(text, age=age)

    # Temperature from free text as fallback
    tf = thermometer_f
    if tf is None:
        tf = _parse_temperature(text)

    # Likely conditions
    conds = guess_conditions(text, age=age)

    # Advice bundle
    advice: List[str] = [
        "Rest and hydrate (warm fluids can soothe the throat).",
        "Gargle warm salt water (not for young children).",
        "Avoid smoking/smoky places.",
        "Honey/lemon for cough or throat irritation (not for <1 year old).",
        "If symptoms worsen quickly or persist >1 week, seek care.",
    ]
    meds: List[str] = [
        "Paracetamol/acetaminophen for fever/pain (follow label dosing).",
        "Ibuprofen if tolerated (with food); avoid if you have ulcers or kidney problems.",
        "Throat lozenges or sprays for temporary relief (age-appropriate).",
    ]

    # Red-flags
    rf = red_flag_assessment(text)

    return {
        "severity": rf["severity"] if rf["present"] else "none",
        "red_flag": rf["present"],
        "red_flag_banner": rf["banner"] if rf["present"] else "",
        "likely_conditions": [{ "name": n, "p": p } for n, p in conds],
        "advice": advice,
        "meds": meds,
        "notes": [
            f"FEVERPAIN={fp}",
            f"CENTOR={ce}",
            f"TempF={'%.1f' % tf if tf is not None else 'n/a'}",
        ],
        "raw": {
            "feverpain": fp,
            "centor": ce,
            "thermometer_f": tf,
            "heart_rate": heart_rate,
            "meds_taken": meds_taken,
        },
    }

# Re-export for API layer convenience
RED_FLAG_BANNER = RED_FLAG_BANNER
