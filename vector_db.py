# Import the Qdrant client to communicate with the Qdrant vector database.
import os

from qdrant_client import QdrantClient

# Import models used to configure the collection and represent stored data.
from qdrant_client.models import VectorParams, Distance, PointStruct


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

    def search(self, query_vector, top_k: int = 5):
        # Search for the points whose vectors are most similar
        # to the user's query vector.
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
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