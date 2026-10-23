## HOW TO RUN
1. uv run uvicorn main:app
2. npx inngest-cli@latest dev http://127.0.0.1:8000/api/inngest --no-discovery

## Gemini embeddings
Set `GEMINI_API_KEY` in the environment or in a local `.env` file before
starting the app. Embeddings use `gemini-embedding-001` with a configured output
size of 768 dimensions.

The app stores these vectors in the `docs_gemini_768` Qdrant collection. Existing
vectors in the `docs` collection are left untouched. Re-ingest documents to
populate the new collection before querying them.