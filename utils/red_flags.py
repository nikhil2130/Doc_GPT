import re

# CDC-style emergency patterns
EMERGENCY_PATTERNS = [
    r"\btrouble\s+breath(ing)?\b",
    r"\bdifficulty\s+breath(ing)?\b",
    r"\bshort(ness)?\s+of\s+breath\b",
    r"\b(chest\s+pain|chest\s+pressure)\b",
    r"\bnew\s+confusion\b",
    r"\binability\s+to\s+wake\b|\b(can'?t|cannot)\s+wake\b",
    r"\bstay\s+awake\b",
    r"\b(blue|bluish|pale|gray|grey)\s+(lips|skin|face|nail\s*beds?)\b",
]

RED_FLAG_MSG = (
    "⚠️ Emergency warning signs detected.\n"
    "Consider urgent evaluation right now. If you are experiencing any of these symptoms, call local emergency services."
)

def detect_red_flags(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in EMERGENCY_PATTERNS)
