from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CHROMA_PATH = ROOT / "data" / "indexes"
COLLECTION = "doc_gpt"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_raw() -> List[Dict[str, Any]]:
    out = []
    for p in RAW.glob("*.json"):
        j = json.loads(p.read_text(encoding="utf-8"))
        out.append(j)
    return out

def chunk_text(text: str, size=800, overlap=120) -> List[str]:
    text = " ".join(text.split())
    chunks, i = [], 0
    while i < len(text):
        chunk = text[i:i+size]
        chunks.append(chunk)
        i += size - overlap
    return chunks

def main():
    docs = load_raw()
    if not docs:
        print("No raw docs found. Run scripts/crawl_clean.py first.")
        return

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(
        name=COLLECTION,
        embedding_function=SentenceTransformerEmbeddingFunction(model_name=MODEL)
    )

    passages, metadatas, ids = [], [], []
    counter = 0
    for d in tqdm(docs, desc="Batches"):
        url = d.get("url", "")
        title = d.get("title", "")
        for ch in chunk_text(d.get("text", "")):
            passages.append(ch)
            metadatas.append({"url": url, "title": title, "snippet": ch[:200]})
            ids.append(f"{counter:08d}")
            counter += 1

    col.add(ids=ids, documents=passages, metadatas=metadatas)
    print(f"Indexed {len(passages)} chunks ({len(docs)} pages).")
    print("CHROMA_PATH =", CHROMA_PATH.resolve())
    print("Collections now present:", [c.name for c in client.list_collections()])

if __name__ == "__main__":
    main()
