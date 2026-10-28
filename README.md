## HOW TO RUN
1. Start Qdrant: `docker run -d --name qdrantRagDb -p 6333:6333 -v "./qdrant_storage:/qdrant/storage" qdrant/qdrant`
2. Start the API: `uv run uvicorn main:app --reload --port 8000`
3. Start the Streamlit app in another terminal: `uv run streamlit run app.py`
4. To use the existing Inngest workflows, run `npx inngest-cli@latest dev http://127.0.0.1:8000/api/inngest --no-discovery`

The app uses `http://127.0.0.1:8000` by default. Set `RAG_API_URL` to use a
different API address. The direct API limits each client to 30 requests per
minute, accepts PDFs up to 20 MB, and runs at most two ingestion/query jobs
concurrently. Rate-limited requests receive HTTP 429; requests waiting over
10 seconds for an available job slot receive HTTP 503.

## NVIDIA Nemotron answers and Gemini embeddings
Set `NVIDIA_API_KEY` in the environment or in a local `.env` file to generate
answers with `nvidia/nemotron-3-ultra-550b-a55b` through NVIDIA's OpenAI-compatible
API. Set `GEMINI_API_KEY` as well; PDF indexing and question search still use
Gemini's `gemini-embedding-001` embedding model with an output size of 768
dimensions. Nemotron is used for answer generation and does not replace the
embedding model.

The app stores these vectors in the `docs_gemini_768` Qdrant collection. Existing
vectors in the `docs` collection are left untouched. Re-ingest documents to
populate the new collection before querying them.

## Chat session isolation and cleanup
Each Streamlit page session gets a new ID. Uploaded vectors are tagged with that
ID, and queries only retrieve vectors from the current session. Vectors belonging
to inactive sessions are deleted by the API cleanup task after
`RAG_SESSION_TTL_SECONDS` (default: 3600 seconds). Cleanup runs every
`RAG_SESSION_CLEANUP_INTERVAL_SECONDS` (default: 300 seconds), starting after
the first interval rather than during API startup. The API creates Qdrant
indexes for the session and expiry fields at startup, and refreshes session
expiry after returning a successful answer. Set both variables to positive
integers if you need different retention and cleanup intervals.
