from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768
EMBED_COLLECTION = "docs_gemini_768"

splitter = SentenceSplitter(
    chunk_size=1000,  # Maximum number of characters in each text chunk.
    chunk_overlap=200  # Number of overlapping characters between consecutive chunks.
)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)  # Load the PDF and extract its text content.
    texts = [d.text for d in docs if getattr(d, "text", None)]  # Extract text from each document, ignoring any without text.
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))  # Split each text into smaller chunks based on the defined chunk size and overlap.   
    return chunks  

def embed_texts(texts: list[str]) -> list[list[float]]:
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
    )
    return [e.values for e in result.embeddings] # Extract and return the embedding vectors
