import asyncio
import logging
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import inngest
import inngest.fast_api
from google import genai
from google.genai import types
from dotenv import load_dotenv
from data_loader import EMBED_COLLECTION, EMBED_DIM, load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult
from rate_limiting import RequestRateLimiter

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
rate_limiter = RequestRateLimiter(requests=30, window_seconds=60)
processing_semaphore = asyncio.Semaphore(2)
MAX_PDF_BYTES = 20 * 1024 * 1024

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


def _load_pdf(pdf_path: str, source_id: str) -> RAGChunkAndSrc:
    chunks = load_and_chunk_pdf(pdf_path)
    return RAGChunkAndSrc(chunks=chunks, source_id=source_id)


def _upsert_pdf(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
    chunks = chunks_and_src.chunks
    source_id = chunks_and_src.source_id
    vecs = embed_texts(chunks)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
    payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
    QdrantStorage(collection=EMBED_COLLECTION, dim=EMBED_DIM).upsert(ids, vecs, payloads)
    return RAGUpsertResult(ingested=len(chunks))


def _search(question: str, top_k: int = 5) -> RAGSearchResult:
    query_vec = embed_texts([question])[0]
    store = QdrantStorage(collection=EMBED_COLLECTION, dim=EMBED_DIM)
    found = store.search(query_vec, top_k)
    return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])


def _generate_answer(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"- {context}" for context in contexts)
    user_content = (
        "Use the following context to answer the question. \n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )
    response = gemini_client.models.generate_content(
        model="gemini-3.8-flash",
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction="You answer questions using only the provided context.",
            max_output_tokens=1024,
            temperature=0.2,
        ),
    )
    if not response.text:
        raise ValueError("Gemini returned an empty answer.")
    return response.text.strip()


def _ingest_pdf(pdf_path: str, source_id: str) -> RAGUpsertResult:
    return _upsert_pdf(_load_pdf(pdf_path, source_id))


def _answer_question(question: str, top_k: int) -> RAGQueryResult:
    found = _search(question, top_k)
    answer = _generate_answer(question, found.contexts)
    return RAGQueryResult(
        answer=answer,
        sources=found.sources,
        num_contexts=len(found.contexts),
    )


async def _run_throttled(func, *args):
    try:
        await asyncio.wait_for(processing_semaphore.acquire(), timeout=10)
    except TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail="The service is busy. Please try again shortly.",
            headers={"Retry-After": "10"},
        ) from exc
    try:
        return await asyncio.to_thread(func, *args)
    finally:
        processing_semaphore.release()


async def enforce_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    retry_after = rate_limiter.retry_after(client_ip)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later.",
            headers={"Retry-After": str(retry_after)},
        )


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)

async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        return _load_pdf(pdf_path, source_id)

    async def _load_step() -> RAGChunkAndSrc:
        return await asyncio.to_thread(_load, ctx)

    async def _upsert_step() -> RAGUpsertResult:
        return await asyncio.to_thread(_upsert_pdf, chunks_and_src)

    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        _load_step,
        output_type=RAGChunkAndSrc,
    )
    ingested = await ctx.step.run(
        "embed-and-upsert",
        _upsert_step,
        output_type=RAGUpsertResult,
    )
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)

async def rag_query_pdf_ai(ctx: inngest.Context):
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    async def _search_step() -> RAGSearchResult:
        return await asyncio.to_thread(_search, question, top_k)

    found = await ctx.step.run(
        "embed-and-search",
        _search_step,
        output_type=RAGSearchResult,
    )

    async def _generate_answer_step() -> str:
        return await asyncio.to_thread(_generate_answer, question, found.contexts)

    answer = await ctx.step.run(
        "llm-answer",
        _generate_answer_step,
        output_type=str,
    )

    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])


@app.post("/api/query", dependencies=[Depends(enforce_rate_limit)])
async def query_documents(query: QueryRequest) -> RAGQueryResult:
    question = query.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be blank.")
    return await _run_throttled(_answer_question, question, query.top_k)


@app.post("/api/ingest", dependencies=[Depends(enforce_rate_limit)])
async def ingest_document(request: Request) -> RAGUpsertResult:
    filename = request.headers.get("x-filename", "")
    source_id = Path(filename).name
    if not source_id.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF file.")

    content_type = request.headers.get("content-type", "")
    if content_type.split(";", maxsplit=1)[0].strip().lower() != "application/pdf":
        raise HTTPException(status_code=415, detail="Content-Type must be application/pdf.")

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = temp_file.name
            size = 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > MAX_PDF_BYTES:
                    raise HTTPException(status_code=413, detail="PDF files must be 20 MB or smaller.")
                temp_file.write(chunk)

        with open(temp_path, "rb") as pdf_file:
            if not pdf_file.read(5).startswith(b"%PDF-"):
                raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")

        return await _run_throttled(_ingest_pdf, temp_path, source_id)
    finally:
        if temp_path is not None:
            os.unlink(temp_path)
