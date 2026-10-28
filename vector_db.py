# Import the Qdrant client to communicate with the Qdrant vector database.
import os
import time

from qdrant_client import QdrantClient

# Import models used to configure the collection and represent stored data.
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    PointStruct,
    PayloadSchemaType,
    Range,
    VectorParams,
)


class QdrantStorage:

    def __init__(self, url=None, collection="docs_gemini_768", dim=768):
        # Use environment variables in production, with localhost as the fallback.
        url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")

        # Connect to local Qdrant or Qdrant Cloud.
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=30,
        )

        self.collection = collection

        # Create the collection if it doesn't exist.
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            )
        else:
            # Check that the existing collection has the expected dimensions.
            collection_info = self.client.get_collection(self.collection)
            vectors = collection_info.config.params.vectors

            if isinstance(vectors, VectorParams) and vectors.size != dim:
                raise ValueError(
                    f"Collection '{self.collection}' has vector size "
                    f"{vectors.size}, but the embedding model produces {dim}."
                )

    def upsert(self, ids, vectors, payloads):
        # Build a list of Qdrant points.
        # Each point contains:
        # - id: a unique identifier for the point
        # - vector: the numerical embedding representing the document
        # - payload: additional information, such as text and source filename
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i]
            )
            for i in range(len(ids))
        ]

        # Insert the points into Qdrant.
        # If an ID already exists, its stored point is replaced/updated.
        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    def ensure_session_indexes(self) -> None:
        indexed_fields = self.client.get_collection(self.collection).payload_schema or {}
        for field_name, schema in (
            ("session_id", PayloadSchemaType.KEYWORD),
            ("session_expires_at", PayloadSchemaType.INTEGER),
        ):
            if field_name not in indexed_fields:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                )

    def search(self, query_vector, session_id: str, top_k: int = 5):
        session_filter = Filter(
            must=[FieldCondition(key="session_id", match={"value": session_id})]
        )
        # Search for the points whose vectors are most similar
        # to the user's query vector.
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=session_filter,
            with_payload=True,  # Include the stored metadata in the results.
            limit=top_k         # Return at most 5 results by default.
        ).points

        # These lists will contain the retrieved document text and sources.
        contexts = []
        sources = set()  # A set automatically removes duplicate source names.

        # Process each matching point returned by Qdrant.
        for result in results:

            # Get the payload, or use an empty dictionary if it is missing.
            payload = result.payload or {}

            # Retrieve the document text and its source from the payload.
            text = payload.get("text", "")
            source = payload.get("source", "")

            # Only include documents that actually contain text.
            if text:
                contexts.append(text)

                # Store the source without duplicates.
                if source:
                    sources.add(source)

        # Return the retrieved text and source names to the caller.
        # These contexts can later be passed to an LLM as RAG context.
        return {
            "contexts": contexts,
            "sources": list(sources)
        }

    def touch_session(self, session_id: str, ttl_seconds: int) -> None:
        self.client.set_payload(
            collection_name=self.collection,
            payload={"session_expires_at": int(time.time()) + ttl_seconds},
            points=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match={"value": session_id},
                        )
                    ]
                )
            ),
        )

    def delete_expired_sessions(self, now: int | None = None) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="session_expires_at",
                            range=Range(lte=now if now is not None else int(time.time())),
                        )
                    ]
                )
            ),
        )