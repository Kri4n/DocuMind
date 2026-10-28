import json
import os
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st


API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 120


def post_json(path: str, payload: dict) -> dict:
    request = Request(
        f"{API_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.load(response)


def ingest_pdf(filename: str, content: bytes, session_id: str) -> dict:
    request = Request(
        f"{API_URL}/api/ingest",
        data=content,
        headers={
            "Content-Type": "application/pdf",
            "X-Filename": filename,
            "X-Session-ID": session_id,
        },
        method="POST",
    )
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.load(response)


def show_api_error(error: HTTPError) -> None:
    try:
        detail = json.loads(error.read()).get("detail", str(error))
    except (json.JSONDecodeError, UnicodeDecodeError):
        detail = str(error)
    st.error(f"Request failed ({error.code}): {detail}")


st.set_page_config(page_title="DocuMind", page_icon="🧠")
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.title("DocuMind 🧠")
st.caption("Upload a PDF to add it to the knowledge base, then ask questions about your documents.")

with st.sidebar:
    st.subheader("Add a document")
    uploaded_pdf = st.file_uploader("Choose a PDF", type=["pdf"])
    if st.button("Ingest PDF", disabled=uploaded_pdf is None):
        try:
            with st.spinner("Reading and indexing the PDF..."):
                result = ingest_pdf(
                    uploaded_pdf.name,
                    uploaded_pdf.getvalue(),
                    st.session_state.session_id,
                )
            st.success(f"Indexed {result['ingested']} text chunks.")
        except HTTPError as error:
            show_api_error(error)
        except URLError as error:
            st.error(f"Could not reach the RAG API at {API_URL}: {error.reason}")

st.subheader("Ask a question")
question = st.text_area("Question", max_chars=2000, placeholder="What does the document say about...")
top_k = st.slider("Context passages", min_value=1, max_value=10, value=5)

if st.button("Ask", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Searching your documents and generating an answer..."):
            result = post_json(
                "/api/query",
                {
                    "question": question.strip(),
                    "top_k": top_k,
                    "session_id": st.session_state.session_id,
                },
            )
        st.markdown(result["answer"])
        if result["sources"]:
            st.caption("Sources: " + ", ".join(result["sources"]))
        if not result["num_contexts"]:
            st.info("No matching passages were found. Ingest a document or try another question.")
    except HTTPError as error:
        show_api_error(error)
    except URLError as error:
        st.error(f"Could not reach the RAG API at {API_URL}: {error.reason}")
