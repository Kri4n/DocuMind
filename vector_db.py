# Import the Qdrant client to communicate with the Qdrant vector database.
from qdrant_client import QdrantClient

# Import models used to configure the collection and represent stored data.
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:

    def __init__(self, url="http://localhost:6333", collection="docs", dim=768):
        # Connect to the Qdrant server.
        # localhost:6333 is the default HTTP address when Qdrant runs locally.
        self.client = QdrantClient(url=url, timeout=30)

        # Save the collection name so other methods can reuse it.
        # A collection is similar to a table in a relational database.
        self.collection = collection

        # Check if the collection already exists.
        # We don't want to create it again if it already exists.
        if not self.client.collection_exists(self.collection):

            # Create a new collection to store document vectors.
            self.client.create_collection(
                collection_name=self.collection,

                # Configure the vectors that this collection will store.
                vectors_config=VectorParams(
                    size=dim,                 # Number of values in each vector.
                    distance=Distance.COSINE  # Compare vectors using cosine similarity.
                )
            )
        else:
            collection = self.client.get_collection(self.collection)
            vectors = collection.config.params.vectors
            if isinstance(vectors, VectorParams) and vectors.size != dim:
                raise ValueError(
                    f"Qdrant collection '{self.collection}' has vector size "
                    f"{vectors.size}, but the embedding model produces {dim}. "
                    "Recreate the collection and re-ingest its documents."
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