# D:\projects\Doc_GPT\utils\followups.py
from __future__ import annotations
from typing import List

def follow_ups_for(suspected_conditions: List[str]) -> List[str]:
    qs: List[str] = []
    joined = " ".join(suspected_conditions).lower()

    if "sore throat" in joined:
        qs += [
            "Do you have cough or runny nose (coryza)?",
            "Did the symptoms start within the last 3 days?",
            "Do you notice white pus/exudate on the tonsils?",
            "Have you had a measured fever in the last 24 hours?",
            "Are the neck glands tender?"
        ]
    if "chest pain" in joined:
        qs += [
            "Is the chest pain severe or spreading to your arm/jaw/back?",
            "Are you short of breath, sweaty, dizzy, or feeling very unwell?",
            "Do you have a history of heart or lung conditions?"
        ]
    if "breathlessness" in joined:
        qs += [
            "Can you speak full sentences or do you need pauses to breathe?",
            "Do you have a reliever inhaler and did it help?",
            "Do you feel chest tightness or wheeze?"
        ]
    if "gastroenteritis" in joined:
        qs += [
            "Are you able to keep fluids down?",
            "How many times have you vomited today?",
            "Is there blood or black material in vomit or stool?"
        ]

    if not qs:
        qs = ["What symptom is bothering you the most right now?",
              "Do any symptoms wake you at night or rapidly worsen?"]

    return qs
