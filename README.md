# Company Website RAG Chatbot

Cost-conscious FastAPI RAG service for the supplied company PDFs and TXT data. During ingestion, it uses `BAAI/bge-small-en-v1.5` locally to create semantic chunks: it starts at paragraph/sentence boundaries, detects topic changes from adjacent embedding similarity, and keeps chunks bounded at about 850-1,100 characters with a one-unit overlap. It then stores normalized embeddings in a compact JSON index. Each request embeds only the question locally, uses cosine retrieval, and calls Groq only for grounded answer writing. A Groq timeout, outage, missing key, or other failure returns a deterministic answer built from retrieved content.

Retrieval combines a local persistent Chroma vector database with BM25 keyword matching, then uses reciprocal-rank fusion and duplicate suppression. FastEmbed creates vectors locally and Chroma stores/query-indexes them using cosine distance; no document text or embedding is sent to a third party. `storage/index.json` holds only chunk metadata, while vectors live in `storage/chroma/`. Document titles and labelled page sections are included during indexing, so direct topic questions (for example, “audience segments”) retrieve the relevant section openings as well as semantically similar content.

The hybrid retriever first fuses Chroma dense results with BM25 keyword results, then uses a local MiniLM cross-encoder to rerank the top 18 candidates. It sends the final `RETRIEVAL_TOP_K=5` chunks to the LLM (set it to `6` in `.env` if needed).

## Evaluation

With FastAPI running locally, evaluate the bot against the curated cases with deterministic retrieval metrics and Groq as an LLM judge:

```sh
python evaluate.py --endpoint http://127.0.0.1:8000/v1/chat
```

It writes `evaluation_report.json` with source-level precision@K, recall@K, MRR, LLM-judged faithfulness/relevance (1-5), and overall interaction success rate. Add reviewed visitor questions and their relevant source files to `evaluation_cases.json` before using scores as launch criteria.

Before retrieval, Groq classifies each validated visitor message as `rag`, `escalation`, `support`, or `career`. Visitors are treated as current or prospective clients by default. Career-related messages receive a recruitment handoff without a document search; escalation and support requests also bypass RAG. Classification uses a short, deterministic JSON response and falls back to RAG if Groq is unavailable.

## Run locally

1. From this directory (which already contains the supplied `.pdf` / `.txt` files):

   ```sh
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   # First setup only: downloads the model into storage/fastembed.
python scripts/ingest.py --source . --output storage/index.json --download-model
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

2. Test it:

   ```sh
   curl -X POST http://localhost:8000/v1/chat -H 'content-type: application/json' \
     -d '{"message":"What services does the company provide?"}'
   ```

`GROQ_API_KEY` is optional for test and outage mode: without it, every supported request receives the retrieval-only fallback.

The Streamlit test frontend uses Docker's proxy URL (`http://localhost:8080/api/chat/chat`) by default. If FastAPI is running directly with Uvicorn instead, launch Streamlit with:

```sh
CHAT_API_URL=http://localhost:8000/v1/chat streamlit run frontend/streamlit_app.py
```

## Deploy with Docker

The supplied source files are already included in the Docker build context. Set the key in `.env`, then run:

```sh
export PATH="/Applications/Docker.app/Contents/Resources/bin:$PATH"
which docker-credential-desktop

docker compose up --build -d
```

Run ingestion again whenever the chunking or retrieval metadata code changes.

The Docker build performs the one-time model download and indexing. At runtime, the API uses the cached local model and does not need Hugging Face access.

The public proxy is at `http://localhost:8080/api/chat/chat`; the container API itself is not published. Rebuild after changing documents with `docker compose build chatbot --no-cache && docker compose up -d`.

## WordPress reverse proxy

Put the application behind the same HTTPS domain as WordPress. Add [wordpress/nginx-location.conf](wordpress/nginx-location.conf) inside that site's Nginx `server` block (adjust upstream host/port as needed), and define `chat_api` globally as shown in [nginx/chatbot.conf](nginx/chatbot.conf). The frontend calls the same-origin endpoint:

```js
fetch('/api/chat/chat', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message})})
```

No API key belongs in PHP, JavaScript, or WordPress settings.

## Security and operating limits

- Request validation: 1,000-character questions, 16 KB body cap, prompt-injection and obvious sensitive-data rejection.
- Output validation redacts accidental key-like values, limits output length, and only answers from retrieved company content.
- Two rate-limit layers: Nginx (10/min/IP with burst 10) and FastAPI (30/min/IP). In multi-container deployments, replace the FastAPI in-memory limiter with a shared Redis limiter.
- `temperature=0.1`, `top_p=0.3`, fixed short output, no chat history by default; this keeps Groq responses predictable and cheap.
- Responses cite document and page metadata. Set a score threshold to refuse weak matches.

For production, terminate TLS at your existing WordPress proxy, set exact `ALLOWED_ORIGINS`, put the service on a private network, and send only health/error metrics (never questions or document text) to monitoring.


