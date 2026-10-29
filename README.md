# legal-ai

A small API for working with legal PDFs: **ask questions, summarise and draft**, with a page citation on every statement. Run it against a local model (Ollama) or a hosted one (Groq).

A live, no-install version of the same idea runs at https://legal.pravija.ai (source: the `pravija-legal-demo` Worker project).

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # choose the model provider, set API_KEY
set -a && source .env && set +a
uvicorn src.app:app --reload
```

Interactive docs: http://localhost:8000/docs. With Docker: `docker build -t legal-ai . && docker run -p 8000:8000 --env-file .env legal-ai`.

**Models.** `LLM_PROVIDER` is `cloudflare` (Workers AI over its REST API), `groq`, or `ollama` (local). With `EMBEDDINGS=cloudflare`, retrieval combines keyword (BM25) and semantic (embedding) search; if the embedding call fails it falls back to keyword search automatically.

**Access.** If `API_KEY` is set, every call except `/health` must send it as `X-API-Key`. Always set it on a public host: it stops strangers from spending your model budget.

## Deploy on Render

1. Render dashboard: **New + > Blueprint**, pick this repo (`render.yaml` is picked up).
2. Enter `CF_ACCOUNT_ID` and `CF_API_TOKEN` when asked. The token is created at dash.cloudflare.com/profile/api-tokens from the "Workers AI" template.
3. After the first deploy, copy the generated `API_KEY` from the service's Environment tab. You need it in the website Worker and in legal-ui.
4. Check `https://<your-service>.onrender.com/health`.

The free plan sleeps after 15 minutes idle and takes about a minute to wake; use a paid instance for client demos. Sessions are held in memory, so a restart clears uploaded documents.

## API

| Call | What it does |
|---|---|
| `POST /upload` (multipart `file`) | Reads the PDF and returns a `session_id`, page count and passage count |
| `POST /query` `{session_id, question}` | Answer from the most relevant passages, plus the passages (with page numbers) as `sources` |
| `POST /summary` `{session_id}` | Structured summary; long documents are condensed section by section first |
| `POST /draft` `{session_id, kind}` | `case_note`, `client_email`, `memo` or `negotiation_note` (see `GET /drafts`) |
| `DELETE /session/{id}` | Remove the document from memory |
| `GET /health` | Status and provider |

Every upload gets its own session that expires after an hour (`SESSION_TTL_SECONDS`), and documents are kept only in memory.

## What changed in v3

- **Cloudflare Workers AI provider** (chat and embeddings), alongside Groq and Ollama.
- **Hybrid retrieval** (keyword + semantic, reciprocal rank fusion) with automatic fallback.
- **API key** protection and a Render blueprint, so a website can call it safely.

## What changed in v2

- **Per-user sessions.** v1 kept one global "latest document", so a second upload replaced the first for every user. Now each upload has its own session.
- **Citations.** Passages keep their page numbers; answers must cite `[p. N]` and the API returns the source text.
- **Drafting** endpoint added, limited to a fixed list of draft types.
- **Real generation limits.** v1 passed `max_tokens`/penalties that Ollama ignores and a 100-token cap, so answers were cut short. Now `num_predict` and `num_ctx` are set properly, and settings come from environment variables.
- **Long documents.** v1 sent the whole PDF to the model in one prompt. Now it fits the context or is summarised in batches.
- **Retrieval** is BM25 keyword search, which works well on legal wording and needs no model download. This removed the heavy `sentence_transformers`, `faiss`, `unstructured` and `langchain` dependencies and the non-pip packages that were listed in requirements (`poppler-utils`, `tesseract-ocr`, `libreoffice`).
- Input checks (PDF only, size and page limits, encrypted or scanned files rejected with a clear message), a proper 502 when the model is down, CORS limited to listed origins.
- Tests (`pytest`, model mocked) and CI.

## Limits

- Scanned PDFs need OCR first; this version reads text PDFs only.
- Sessions live in one process's memory. For several workers or restarts, swap `Store` in `src/service.py` for Redis.
- Not legal advice; a lawyer should review outputs.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
