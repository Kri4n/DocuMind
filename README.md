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

## Gemini embeddings
Set `GEMINI_API_KEY` in the environment or in a local `.env` file before
starting the app. Embeddings use `gemini-embedding-001` with a configured output
size of 768 dimensions.

The app stores these vectors in the `docs_gemini_768` Qdrant collection. Existing
vectors in the `docs` collection are left untouched. Re-ingest documents to
populate the new collection before querying them.
