# utils/llm_client.py
from __future__ import annotations
import os, json
import httpx
from typing import List, Dict, Optional

# Defaults target LM Studio’s local server
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_MODEL       = os.getenv("LLM_MODEL", "meta-llama-3.1-8b-instruct")
TIMEOUT         = float(os.getenv("LLM_TIMEOUT", "60"))

HEADERS = {
    # LM Studio does not require a key by default, but some builds check header presence
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY','no-key')}",
}

def chat_complete(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """
    Minimal OpenAI-compatible /v1/chat/completions call against LM Studio.
    Returns assistant 'content' (string). Raises on HTTP errors.
    """
    url = f"{OPENAI_BASE_URL}/chat/completions"
    payload = {
        "model": model or LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.post(url, headers=HEADERS, json=payload)
        r.raise_for_status()
        data = r.json()
    try:
        return data["choices"][0]["message"]["content"] or ""
    except Exception:
        return json.dumps(data, ensure_ascii=False)
