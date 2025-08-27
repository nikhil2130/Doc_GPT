# D:\projects\Doc_GPT\utils\answer_style.py
from __future__ import annotations

import os
import re
from typing import List, Dict, Any, Optional

# -------------------------
# Clarifying questions
# -------------------------
KEYWORD_QUESTIONS = [
    (r"\b(headache|migraine)\b", [
        "When did it start and how long does each episode last?",
        "Where is the pain (one side, both sides, behind eyes)?",
        "Any fever, stiff neck, confusion, new weakness, vision changes, or head injury?",
        "Any nausea, vomiting, sensitivity to light or sound?",
        "Are you pregnant, or do you have high blood pressure?",
    ]),
    (r"\b(sore\s*throat|throat)\b", [
        "Any fever, difficulty swallowing, drooling, or trouble breathing?",
        "Any recent exposure to someone ill, or recent travel?",
        "Do you have cough, runny nose, or swollen neck glands?",
    ]),
    (r"\b(chest\s*pain|tightness)\b", [
        "Does it worsen with exertion and improve with rest?",
        "Any shortness of breath, sweating, nausea, or pain radiating to jaw/arm/back?",
        "Any risk factors (age >40, smoking, diabetes, high BP, high cholesterol)?",
    ]),
    (r"\b(short(ness)?\s*of\s*breath|breath(ing)?\s*difficulty)\b", [
        "Did it start suddenly or gradually?",
        "Any chest pain, wheeze, cough, fever, leg swelling, or recent long travel?",
        "Do you have asthma/COPD or heart disease?",
    ]),
    (r"\b(abdominal|stomach)\s*pain\b", [
        "Where exactly is the pain and does it move?",
        "Any vomiting, diarrhea/constipation, blood in stool, or fever?",
        "Does food make it better or worse?",
    ]),
]

def build_clarifying_questions(user_text: str) -> List[str]:
    t = (user_text or "").lower()
    qs: List[str] = []
    for pattern, questions in KEYWORD_QUESTIONS:
        if re.search(pattern, t):
            qs.extend(questions)
    # De-dup and limit
    out, seen = [], set()
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[:6]

# -------------------------
# Extractive synthesis
# -------------------------
def _top_sentences(passages: List[Dict[str, Any]], max_chars: int = 1200) -> str:
    """
    Take the top passages as-is and keep the first 3-5 short sentences that look like advice,
    warnings, or definitions. This stays LLM-free & cite-able.
    """
    # Simple sentence splitter by punctuation (robust enough for NHS/CDC text)
    def split_sents(text: str) -> List[str]:
        # Normalize whitespace and split
        s = re.sub(r"\s+", " ", text.strip())
        parts = re.split(r"(?<=[\.\!\?])\s+", s)
        return [p.strip() for p in parts if p.strip()]

    # Score: prefer sentences that include verbs like "call", "go to", "seek", "take", "drink", "rest"
    verbs = re.compile(r"\b(call|go|seek|take|use|drink|rest|avoid|is|are|should|contact|see|stay|keep|rinse|gargle)\b", re.I)

    collected: List[str] = []
    for p in passages:
        for sent in split_sents(p["text"])[:8]:
            if len(sent) < 30:
                continue
            score = 1 + (1 if verbs.search(sent) else 0)
            if "emergency" in sent.lower() or "immediate" in sent.lower():
                score += 1
            collected.append((score, sent))

    # Sort by score then keep until char budget
    collected.sort(key=lambda x: (-x[0], len(x[1])))
    chosen: List[str] = []
    total = 0
    for _, s in collected:
        if s in chosen:
            continue
        if total + len(s) > max_chars:
            break
        chosen.append(s)
        total += len(s)
        if len(chosen) >= 8:
            break
    return "\n- " + "\n- ".join(chosen) if chosen else ""

def _pick_bullets(passages: List[Dict[str, Any]], keywords: List[str]) -> List[str]:
    bullets: List[str] = []
    pat = re.compile("|".join([re.escape(k) for k in keywords]), re.I)
    for p in passages:
        lines = re.split(r"[•\-\n\r]+", p["text"])
        for ln in lines:
            s = re.sub(r"\s+", " ", ln).strip()
            if len(s) > 40 and pat.search(s):
                bullets.append(s)
    # unique + limit
    out, seen = [], set()
    for b in bullets:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out[:6]

def clinician_style_answer(
    user_question: str,
    user_profile: str,
    passages: List[Dict[str, Any]],
    show_clarifiers: bool,
    red_flag_detected: bool,
) -> str:
    clarifiers = build_clarifying_questions(user_question) if show_clarifiers else []

    # pull likely causes/self-care from common phrases
    likely = _pick_bullets(passages, ["may be caused", "common causes", "could be", "symptoms include"])
    care = _pick_bullets(passages, ["self care", "at home", "you can", "paracetamol", "ibuprofen", "rest", "fluids", "salt water", "gargle", "honey"])
    urgent = _pick_bullets(passages, ["emergency", "urgent", "A&E", "999", "call emergency", "seek immediate"])

    # fallback: plain extractive snippet list
    extractive = _top_sentences(passages)

    # build answer
    lines: List[str] = []

    # 0) Opening
    opening = "Here’s a careful, clinician‑style summary from reputable sources."
    if user_profile:
        opening += f" (Context: {user_profile})"
    lines.append(opening)

    # 1) Clarifying Qs (brief)
    if clarifiers:
        lines.append("\n**Before advice, a few quick questions to guide safety & accuracy:**")
        for q in clarifiers:
            lines.append(f"- {q}")

    # 2) Likely / possibilities
    if likely:
        lines.append("\n**What it might be (not a diagnosis):**")
        for b in likely:
            lines.append(f"- {b}")

    # 3) Self-care
    if care:
        lines.append("\n**Self‑care you can consider (if no red flags):**")
        for b in care:
            lines.append(f"- {b}")

    # 4) Urgent care
    if red_flag_detected or urgent:
        lines.append("\n**When to seek urgent care:**")
        if red_flag_detected:
            lines.append(f"- {os.getenv('RED_FLAG_OVERRIDE', 'Emergency warning signs detected from your message. ')}{os.getenv('RED_FLAG_SUFFIX', 'If these apply now, seek urgent care (call local emergency services).')}")
        for b in urgent:
            lines.append(f"- {b}")

    # 5) Practical next steps
    lines.append("\n**What to do next:**")
    lines.append("- If symptoms are severe, new, or worsening, seek medical care.")
    lines.append("- If symptoms are mild and no red flags, trial 24–48h of self‑care, then re‑assess.")
    lines.append("- Keep a note of triggers, duration, associated symptoms, and any medicines taken.")

    # 6) Extractive tail to keep the answer grounded in source phrasing
    if extractive:
        lines.append("\n**Key lines from sources:**")
        lines.append(extractive)

    # 7) Disclaimer
    lines.append("\n*I’m not a doctor. This assistant is for information only and can’t examine you. For serious or uncertain problems, see a clinician.*")

    return "\n".join(lines)

# -------------------------
# Optional LLM polish (short, safe rewrite)
# -------------------------
def llm_polish_answer_if_available(
    base_answer: str,
    user_question: str,
    user_profile: str,
    passages: List[Dict[str, Any]],
    red_flag_detected: bool,
) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    # Lightweight, defensive polishing prompt
    system = (
        "You are a cautious clinician writing for laypeople. "
        "Rewrite the user-provided draft into a clear, concise, structured answer with headings: "
        "[Clarifying questions (if any)] [What it might be] [Self-care] [When to seek urgent care] [What to do next] [Disclaimer]. "
        "Keep it short (120–220 words). Never give definitive diagnoses, only possibilities. "
        "Never invent facts not present in the draft. Never contradict safety warnings. "
        "If the draft contains emergency warning signs, repeat them prominently."
    )
    user = (
        f"User question: {user_question}\n"
        f"User profile: {user_profile or 'n/a'}\n\n"
        f"Draft to polish:\n{base_answer}\n"
    )

    try:
        # Use OpenAI Responses API if installed, otherwise fallback to legacy
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        out = resp.choices[0].message.content.strip()
        # extra guard: must include Disclaimer section
        if "Disclaimer" not in out:
            return None
        return out
    except Exception:
        return None
