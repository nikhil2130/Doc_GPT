## Welcome to Doc_GPT!

This guide provides essential context for AI agents working on this medical conversational RAG assistant. Understanding these concepts is key to being productive in this codebase.

### Big Picture: A Medical RAG Pipeline

Doc_GPT is a local-first, multi-stage RAG (Retrieval-Augmented Generation) system designed for medical Q&A. The core architecture consists of three main stages:

1.  **Triage & Safety (`/intake`)**: All user queries first go through a safety check in `api/main.py`. The `red_flag_assessment` function in `utils/triage_rules.py` uses regex patterns from `utils/red_flags.py` to detect urgent medical symptoms. If a "red flag" is found, the system returns a warning and does not proceed to the answer generation stage.

2.  **Retrieval (`/ask`)**: If the query passes the safety check, it moves to the retrieval stage. The system uses a hybrid search approach, combining semantic search (via a flat index created by `scripts/build_flat_index.py`) and keyword search (BM25). The `flat.search` method in `utils/retrieval.py` handles this hybrid search, querying the index located in `data/flatindex/`.

3.  **Generation (`/ask`)**: The retrieved documents are then passed to a large language model (LLM) to generate a concise answer. The `short_answer_from_docs` function in `utils/answer_style.py` constructs the prompt, and the `llm_client.py` module handles communication with the LLM, which can be a local model (like Ollama) or the OpenAI API.

### Key Data Flows

- **Indexing**: The process starts with a list of trusted medical URLs in `data/sources.csv`. The `scripts/crawl_clean.py` script fetches, cleans, and stores the content in `data/raw/`. Then, `scripts/build_flat_index.py` chunks the text, generates embeddings, and creates the search index in `data/flatindex/`.
- **Querying**: A user query from the frontend (`web/index.html`) hits the `/ask` endpoint in `api/main.py`. The query is processed, sent to the retrieval and generation stages, and the final answer, along with citations, is returned to the user.

### Critical Developer Workflows

- **Building the Index**: Before running the backend, you must build the search index. This is a two-step process:

  1.  `python ./scripts/crawl_clean.py`: Crawls and cleans the data.
  2.  `python ./scripts/build_flat_index.py`: Builds the flat index.

- **Running the Backend**: To run the FastAPI server:
  `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`

- **Evaluation**: To evaluate the system's performance, use the `eval_docgpt.py` script:
  `python ./scripts/eval_docgpt.py`
  This script runs queries from `data/eval.jsonl` and evaluates the quality of the generated answers and citations.

### Project-Specific Conventions

- **Hybrid Search**: The project uses a custom hybrid search implementation in `utils/retrieval.py`. It combines a flat index for semantic search with BM25 for keyword matching. Don't assume a standard vector database is being used.
- **Red Flag Detection**: The safety system is a core feature. The regex patterns in `utils/red_flags.py` are critical for detecting urgent medical situations. When modifying triage logic, refer to `utils/triage_rules.py`.
- **Configuration**: The system is configured via a `.env` file at the project root. This file controls the LLM connection, embedding model, and other settings. Refer to the `README.md` for the required environment variables.
- **Local-First LLMs**: The `llm_client.py` is designed to work with local LLMs by default. The `OPENAI_BASE_URL` environment variable is typically pointed to a local server like Ollama or LM Studio.

By understanding these key aspects of the Doc_GPT project, you can navigate the codebase more effectively and make meaningful contributions.
