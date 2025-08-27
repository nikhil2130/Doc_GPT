# D:\projects\Doc_GPT\utils\otc.py
from __future__ import annotations
from typing import List, Optional

def adult_paracetamol() -> str:
    # NHS adult dosing
    return ("Paracetamol (adults): 500 mg x2 tablets (1 g) up to 4 times in 24 h; "
            "leave ≥4 h between doses; max 4 g/day. " 
            "Avoid duplicates in combination products. " 
            "Safe in pregnancy at recommended doses. [NHS]")

def adult_ibuprofen() -> str:
    # NHS adult dosing (OTC range; higher only if prescribed)
    return ("Ibuprofen (adults): 200–400 mg up to 3 times/day with food; "
            "some may require up to 600 mg QID only if prescribed. "
            "Avoid if pregnancy (esp. after 20 weeks), GI ulcers, kidney disease; "
            "do not combine with other NSAIDs. [NHS]")

def otc_suggestions(age: int, gender: str, suspected: List[str]) -> List[str]:
    out: List[str] = []
    if age < 16:
        out.append("OTC medicines for children depend on age/weight — avoid adult doses; seek child‑specific advice.")
        return out

    # Adults
    # Default pain/fever relief
    out.append(adult_paracetamol())
    # Offer ibuprofen unless contraindicated flags are obvious (we don't have full PMH — keep generic caution)
    out.append(adult_ibuprofen())

    # Gentle tailoring
    sus = " ".join(suspected).lower()
    if "gastroenteritis" in sus:
        out.append("Avoid NSAIDs if dehydrated; prefer paracetamol for fever/aches.")
    if "asthma" in sus or "breathless" in sus:
        out.append("If you have asthma, check personal suitability of NSAIDs; some people with asthma are sensitive to NSAIDs.")

    return out
