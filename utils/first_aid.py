# D:\projects\Doc_GPT\utils\first_aid.py
from __future__ import annotations
from typing import List

def steps_for(condition: str) -> List[str]:
    c = condition.lower()
    steps: List[str] = []

    if "sore throat" in c:
        steps += [
            "Rest, drink plenty of fluids; warm drinks may soothe.",
            "Gargle warm salt water (adults/older children only).",
            "Avoid smoking/smoky places.",
            "Use lozenges/ice lollies for comfort (avoid choking hazards in young children).",
        ]
    if "strep" in c:
        steps += [
            "If high FeverPAIN/Centor or advised by clinician, strep testing or antibiotics may be appropriate.",
            "Avoid close contact if febrile; practice hand hygiene."
        ]
    if "breathlessness" in c:
        steps += [
            "Sit upright, loosen tight clothing, focus on slow breaths.",
            "If you have a reliever inhaler (e.g., salbutamol), use as directed.",
            "Seek urgent care if symptoms are severe or worsening."
        ]
    if "chest pain" in c:
        steps += [
            "Stop activity, sit/lie in a comfortable position.",
            "If previously prescribed glyceryl trinitrate (GTN) for angina, use as directed.",
            "If pain is severe, spreading, or with breathlessness/feeling unwell — call emergency services."
        ]
    if "gastroenteritis" in c:
        steps += [
            "Small frequent sips of oral rehydration solution (ORS).",
            "Avoid alcohol/caffeine; introduce bland foods as tolerated.",
            "Seek care if signs of dehydration or blood in stool/vomit."
        ]
    if "flu-like" in c:
        steps += [
            "Rest, fluids; consider paracetamol/ibuprofen for fever and aches.",
            "Self‑isolate if febrile and follow local public health guidance."
        ]

    # Fallback
    if not steps:
        steps = ["General first aid: rest, fluids, monitor symptoms, seek medical advice if worsening."]
    return steps
