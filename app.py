"""
Customer Support Agent — Streamlit RAG chat application.
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from ingest import EMBEDDING_MODEL, ingest_pdfs

load_dotenv()

LLM_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
RETRIEVAL_K = 3

SYSTEM_PROMPT = (
    "You are a helpful customer support agent for the company. "
    "Answer ONLY based on the context provided. "
    "If information is missing, explicitly state that you don't know based on company policy. "
    "Do not make up information or use knowledge outside the provided context.\n\n"
    "Context:\n{context}"
)


def validate_env() -> dict[str, str]:
    """Validate required environment variables."""
    config = {
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "pinecone_api_key": os.getenv("PINECONE_API_KEY", ""),
        "pinecone_index_name": os.getenv("PINECONE_INDEX_NAME", ""),
    }

    missing = [key for key, value in config.items() if not value]
    if missing:
        friendly = {
            "groq_api_key": "GROQ_API_KEY",
            "pinecone_api_key": "PINECONE_API_KEY",
            "pinecone_index_name": "PINECONE_INDEX_NAME",
        }
        labels = [friendly[k] for k in missing]
        st.error(
            f"Missing required environment variables: **{', '.join(labels)}**. "
            "Copy `.env.example` to `.env` and add your API keys."
        )
        st.stop()

    return config


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Connecting to Pinecone...")
def get_vectorstore(index_name: str, api_key: str) -> PineconeVectorStore:
    Pinecone(api_key=api_key)
    embeddings = get_embeddings()
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )


@st.cache_resource(show_spinner="Initializing LLM...")
def get_rag_chain(groq_api_key: str, pinecone_api_key: str, index_name: str):
    llm = ChatGroq(
        model=LLM_MODEL,
        groq_api_key=groq_api_key,
        temperature=0,
    )

    vectorstore = get_vectorstore(index_name, pinecone_api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_ingestion(uploaded_files) -> None:
    """Process uploaded PDFs and push to Pinecone."""
    if not uploaded_files:
        return

    with st.spinner(f"Ingesting {len(uploaded_files)} document(s)..."):
        temp_paths: list[Path] = []
        try:
            for uploaded in uploaded_files:
                suffix = Path(uploaded.name).suffix or ".pdf"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    temp_paths.append(Path(tmp.name))

            chunk_count = ingest_pdfs(temp_paths)
            st.success(f"Ingested {len(uploaded_files)} file(s) — {chunk_count} chunks uploaded.")
            get_vectorstore.clear()
            get_rag_chain.clear()
        except EnvironmentError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")
        finally:
            for path in temp_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass


def main() -> None:
    st.set_page_config(
        page_title="Customer Support Agent",
        page_icon="💬",
        layout="centered",
    )

    init_session_state()
    config = validate_env()

    st.title("Customer Support Agent")
    st.caption("Ask questions about company policies — answers are grounded in uploaded documents.")

    with st.sidebar:
        st.header("Document Ingestion")
        st.markdown(
            "Upload PDF policy documents to add them to the knowledge base."
        )
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if st.button("Ingest Documents", use_container_width=True):
            handle_ingestion(uploaded_files)

        st.divider()
        st.markdown("**Settings**")
        st.text(f"Model: {LLM_MODEL}")
        st.text(f"Embeddings: {EMBEDDING_MODEL}")
        st.text(f"Index: {config['pinecone_index_name']}")
        st.text(f"Retrieval: top {RETRIEVAL_K} chunks")

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    rag_chain = get_rag_chain(
        config["groq_api_key"],
        config["pinecone_api_key"],
        config["pinecone_index_name"],
    )

    render_chat_history()

    if prompt := st.chat_input("How can I help you today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching policy documents..."):
                try:
                    result = rag_chain.invoke({"input": prompt})
                    answer = result.get("answer", "Sorry, I could not generate a response.")
                except Exception as exc:
                    answer = (
                        f"An error occurred while processing your request: {exc}. "
                        "Please verify your API keys and Pinecone index configuration."
                    )
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
