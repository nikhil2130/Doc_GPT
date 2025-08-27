# D:\projects\Doc_GPT\utils\guidelines.py

from __future__ import annotations
import re
from typing import List, Dict

# --- Communication style pillars ---
# Based on CDC Clear Communication Index (plain language, active voice, prioritized info)
# and NICE shared decision-making (options + next steps)
CLINIC_STYLE_SYSTEM = """You are Doc_GPT, a careful clinical information assistant.
You are NOT a doctor, you do not diagnose, and you never give prescriptions.
Your job is to: (1) understand the user's symptoms and context, (2) summarize what reputable
sources say, (3) give practical self‑care and “what to watch for”, and (4) make escalation
advice very clear. Always use plain language and short sentences.

Rules:
- Be empathetic and calm. Avoid fear.
- Use headings and bullets.
- Lead with the most important points first.
- Use the user’s own terms, but define any medical words simply.
- Offer options (“you could try…”, “another option is…”). 
- Add a short safety notice at the end.
- If any emergency warning signs appear, show an URGENT box first.

Tone examples:
- “Here’s a short plan you can try at home…”
- “If any of these happen, get help now.”"""

# --- Emergency warning signs (lightweight, general) ---
EMERGENCY_PATTERNS = [
    r"\btrouble\s+breath(ing)?\b",
    r"\bdifficulty\s+breath(ing)?\b",
    r"\bshort(ness)?\s+of\s+breath\b",
    r"\b(chest\s+pain|chest\s+pressure)\b",
    r"\bnew\s+confusion\b",
    r"\bfaint(ing)?\b",
    r"\b(inability|unable)\s+to\s+(wake|stay\s*awake)\b",
    r"\b(blue|bluish|pale|gray|grey)\s+(lips|skin|face|nails?)\b",
    r"\bstiff\s+neck\b",
    r"\bsevere\s+(headache|abdominal\s+pain|tummy\s+ache)\b",
    r"\bblood(y)?\s+(vomit|stool|cough)\b",
    r"\bpregnan(t|cy)\b.*\bsevere\b.*\b(pain|bleeding)\b",
]

RED_FLAG_MSG = (
    "⚠️ URGENT ADVICE\n"
    "If you have severe trouble breathing, chest pain, new confusion, can’t stay awake, "
    "blue/grey lips or skin, or rapidly worsening symptoms, seek emergency care now "
    "(e.g., local emergency number or nearest ER)."
)

def detect_red_flags(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in EMERGENCY_PATTERNS)

# --- Answer template the LLM fills in ---
CLINIC_ANSWER_SKELETON = """Here’s what reputable sources say about this topic:
{bulleted_findings}

Home care you can try:
{home_care}

What to watch for (get help sooner if these appear):
{watch_fors}

If you need to speak to a clinician (non‑emergency):
- Use your local telehealth/GP/urgent care. Bring a list of symptoms, timing, meds, and allergies.

Notes:
- I’m not a doctor. This is for information only, not a diagnosis. If symptoms are severe or getting worse, seek medical care immediately.
"""

# Helper to format bullets safely from retrieved snippets
def format_bullets(snippets: List[str], max_items: int = 6) -> str:
    out = []
    for s in snippets[:max_items]:
        s = s.strip().replace("\n", " ").strip(" -•")
        if len(s) > 350:
            s = s[:347].rstrip() + "..."
        out.append(f"- {s}")
    return "\n".join(out) if out else "- (no good matches, try rephrasing your question)"

