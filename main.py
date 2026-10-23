import asyncio
import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from google import genai
from google.genai import types
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import EMBED_COLLECTION, EMBED_DIM, load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)

async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage(collection=EMBED_COLLECTION, dim=EMBED_DIM).upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))
        

    async def _load_step() -> RAGChunkAndSrc:
        return await asyncio.to_thread(_load, ctx)

    async def _upsert_step() -> RAGUpsertResult:
        return await asyncio.to_thread(_upsert, chunks_and_src)

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
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage(collection=EMBED_COLLECTION, dim=EMBED_DIM)
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    async def _search_step() -> RAGSearchResult:
        return await asyncio.to_thread(_search, question, top_k)

    found = await ctx.step.run(
        "embed-and-search",
        _search_step,
        output_type=RAGSearchResult,
    )

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question. \n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    def _generate_answer() -> str:
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

    async def _generate_answer_step() -> str:
        return await asyncio.to_thread(_generate_answer)

    answer = await ctx.step.run(
        "llm-answer",
        _generate_answer_step,
        output_type=str,
    )

    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
