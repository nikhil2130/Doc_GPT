import re

# Patterns covering common emergency red-flags
EMERGENCY_PATTERNS = [
    r"\b(trouble|difficulty)\s+breath(ing)?\b",
    r"\bshort(ness)?\s+of\s+breath\b",
    r"\b(chest\s+(pain|pressure))\b",
    r"\bnew\s+confusion\b",
    r"\b(inability|unable|can('?|no)t)\s+to\s+(wake|stay\s+awake)\b",
    r"\b(blue|bluish|pale|gray|grey)\s+(lips?|face|skin|nail\s*beds?)\b",
    r"\b(swelling|swollen)\s+(tongue|throat)\b",
    r"\bwheez(ing)?\b",
    r"\b(hives|raised\s+rash)\b.*\b(breath|throat)\b",
    r"\b(signs?\s+of\s+severe\s+dehydration|no\s+urine\s+for\s+(\d+|several)\s+hours)\b",
    r"\b(stiff\s+neck)\b",
    r"\b(severe|worst)\s+headache\b",
    r"\bpersistent\s+(vomit(ing)?|diarrho(e|a))\b.*\b(blood|black)\b",
    r"\b(pressure|tightness)\s+in\s+chest\b",
    r"\brapid\s+or\s+irregular\s+heartbeat\b",
]

RED_FLAG_BANNER = (
    "⚠️ Emergency warning signs detected.\n"
    "Seek emergency care now (call local emergency number). Examples: "
    "severe chest pain, trouble breathing, new confusion, inability to wake/stay awake, "
    "or blue/gray lips/face."
)

def detect_red_flags(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in EMERGENCY_PATTERNS)

def classify_red_flag_severity(text: str) -> str:
    return "emergency" if detect_red_flags(text) else "none"
