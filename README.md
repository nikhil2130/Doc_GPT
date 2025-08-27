📘 Doc_GPT: Medical Conversational RAG Assistant

Doc_GPT is a local-first medical question–answering assistant built to showcase end-to-end RAG pipelines, LLM integration, and AI-native engineering.
It uses NHS / CDC health pages as trusted data sources, detects emergency "red flag" symptoms, and provides answers with citations for transparency.


🚀 Features

✅ Crawls and cleans trusted medical sources (NHS, CDC, WHO)
✅ Chunks + embeds text with Sentence-Transformers and indexes via Flat Index + BM25 hybrid retrieval
✅ Serves a FastAPI backend (/ask, /healthz)
✅ Provides citations + red flag warnings in answers
✅ Web frontend (/web/index.html) for simple chat interface
✅ Works with local LLMs (Ollama / LM Studio / llama.cpp) or OpenAI API
✅ Evaluation script with sample queries and citation metrics

📂 Project Structure
D:\projects\Doc_GPT
│   requirements.txt
│   .env
│   README.md   ← (this file)
│
├── api/
│   └── main.py              # FastAPI backend
│
├── data/
│   ├── sources.csv          # list of trusted NHS/CDC URLs
│   ├── raw/                 # crawled + cleaned JSON docs
│   └── flatindex/           # embeddings, bm25 cache
│
├── scripts/
│   ├── crawl_clean.py       # crawler + cleaner
│   ├── chunk_index.py       # chromadb indexing (optional)
│   ├── build_flat_index.py  # flat + hybrid index
│   └── eval_docgpt.py       # evaluation runner
│
├── utils/
│   └── red_flags.py         # regex detection of emergency symptoms
│
└── web/
    └── index.html           # simple frontend

⚙️ Setup
1. Clone + create venv
cd D:\projects
git clone <this-repo> Doc_GPT
cd Doc_GPT
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

3. Environment variables (.env file at project root)
# LLM connection
OPENAI_BASE_URL=http://127.0.0.1:1234/v1
OPENAI_API_KEY=not-needed-for-local
LLM_MODEL=meta-llama-3.1-8b-instruct

# Indexing
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
COLLECTION_NAME=doc_gpt
CHROMA_DIR=./data/chroma

🕸️ Crawl Data

Edit data/sources.csv with NHS/CDC links like:

title,url
Sore throat,https://www.nhs.uk/symptoms/sore-throat/
Chest pain,https://www.nhs.uk/symptoms/chest-pain/
Shortness of breath,https://www.nhs.uk/symptoms/shortness-of-breath/


Run crawler:

python .\scripts\crawl_clean.py --max 3 --delay 0.2

📑 Build Index

Flat + BM25 hybrid:

python .\scripts\build_flat_index.py


Check output:

dir .\data\flatindex\

🔥 Run Backend
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000


Then open in browser:

http://127.0.0.1:8000/web/index.html

💬 Example Query
Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method POST `
  -ContentType "application/json" `
  -Body (@{ query = "I have a sore throat and mild fever. What can I do at home?"; k = 6 } | ConvertTo-Json)


Returns JSON with:

answer (LLM composed)

citations (from NHS/CDC docs)

red_flag (emergency symptom detector)

🧪 Evaluation

Run built-in evaluation:

python .\scripts\eval_docgpt.py --host 127.0.0.1 --port 8000 --k 6


Outputs:

Citation@1 rate

Citation ANY rate

Eval report JSON in data/eval_report.json

⚠️ Disclaimer

This project is for educational purposes only.
It is not a substitute for professional medical advice.
Always consult a qualified clinician for health concerns.